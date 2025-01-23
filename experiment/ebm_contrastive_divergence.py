import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from datasets import load_dataset
from torchvision import transforms
import torchvision
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.callbacks import Callback
import numpy as np
from torch import optim
import random
import math


class Sampler:
    """
    In the next part, we look at the training with sampled elements. To use the contrastive divergence objective,
    we need to generate samples during training. Previous work has shown that due to the high dimensionality of
    images, we need a lot of iterations inside the MCMC sampling to obtain reasonable samples. However, there is
    a training trick that significantly reduces the sampling cost: using a sampling buffer. The idea is that we
    store the samples of the last couple of batches in a buffer, and re-use those as the starting point of the
    MCMC algorithm for the next batches. This reduces the sampling cost because the model requires a
    significantly lower number of steps to converge to reasonable samples. However, to not solely rely on
    previous samples and allow novel samples as well, we re-initialize 5% of our samples from scratch
    (random noise between -1 and 1).

    Below, we implement the sampling buffer. The function sample_new_exmps returns a new batch of “fake”
    images. We refer to those as fake images because they have been generated, but are not actually part
    of the dataset. As mentioned before, we use initialize 5% randomly, and 95% are randomly picked from
    our buffer. On this initial batch, we perform MCMC for 60 iterations to improve the image quality and
    come closer to samples from. In the function generate_samples, we implemented the MCMC for images.
    """

    def __init__(self, model, img_shape, sample_size, max_len=8192):
        """
        Inputs:
            model - Neural network to use for modeling E_theta
            img_shape - Shape of the images to model
            sample_size - Batch size of the samples
            max_len - Maximum number of data points to keep in the buffer
        """
        super().__init__()
        self.model = model
        self.img_shape = img_shape
        self.sample_size = sample_size
        self.max_len = max_len
        self.examples = [
            (torch.rand((1,) + img_shape) * 2 - 1) for _ in range(self.sample_size)
        ]
        # init examples are random samples
        # 添加分布式判断
        self.is_distributed = torch.distributed.is_initialized()

    def sample_new_exmps(self, steps=60, step_size=10, device=None):
        """
        Function for getting a new batch of "fake" images.
        Inputs:
            steps - Number of iterations in the MCMC algorithm
            step_size - Learning rate nu in the algorithm above
        """
        # 添加分布式同步
        if self.is_distributed:
            torch.distributed.barrier()
            # 同步缓冲区操作需要更复杂的实现
        # Choose 95% of the batch from the buffer, 5% generate from scratch
        with torch.no_grad():
            n_new = np.random.binomial(self.sample_size, 0.05)
            rand_imgs = torch.rand((n_new,) + self.img_shape) * 2 - 1

            old_imgs = torch.cat(
                random.choices(self.examples, k=self.sample_size - n_new), dim=0
            )
            inp_imgs = torch.cat([rand_imgs, old_imgs], dim=0).detach().to(device)

            # Perform MCMC sampling
        inp_imgs = Sampler.generate_samples(
            self.model, inp_imgs, steps=steps, step_size=step_size
        )

        # Add new images to the buffer and remove old ones if needed
        self.examples = (
            list(inp_imgs.to(torch.device("cpu")).chunk(self.sample_size, dim=0))
            + self.examples
        )
        self.examples = self.examples[: self.max_len]
        return inp_imgs

    @staticmethod
    def generate_samples(
        model, inp_imgs, steps=60, step_size=10, return_img_per_step=False
    ):
        """
        Function for sampling images for a given model.
        Inputs:
            model - Neural network to use for modeling E_theta
            inp_imgs - Images to start from for sampling. If you want to generate new images, enter noise between -1 and 1.
            steps - Number of iterations in the MCMC algorithm.
            step_size - Learning rate nu in the algorithm above
            return_img_per_step - If True, we return the sample at every iteration of the MCMC
        """
        # Before MCMC: set model parameters to "required_grad=False"
        # because we are only interested in the gradients of the input.
        is_training = model.training
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        inp_imgs.requires_grad = True

        # Enable gradient calculation if not already the case
        had_gradients_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

        # We use a buffer tensor in which we generate noise each loop iteration.
        # More efficient than creating a new tensor every iteration.
        noise = torch.randn(inp_imgs.shape, device=inp_imgs.device)

        # List for storing generations at each step (for later analysis)
        imgs_per_step = []

        # Loop over K (steps)
        for _ in range(steps):
            # Part 1: Add noise to the input.
            # sigma = math.sqrt(step_size * 2)
            # noise.normal_(0, sigma)
            noise.normal_(0, 0.005)
            inp_imgs.data.add_(noise.data)
            inp_imgs.data.clamp_(min=-1.0, max=1.0)

            # Part 2: calculate gradients for the current input.
            out_imgs = -model(inp_imgs)
            out_imgs.sum().backward()
            inp_imgs.grad.data.clamp_(-0.03, 0.03) # For stabilizing and preventing too high gradients

            # Apply gradients to our current samples
            inp_imgs.data.add_(-step_size * inp_imgs.grad.data)
            inp_imgs.grad.detach_()
            inp_imgs.grad.zero_()
            inp_imgs.data.clamp_(min=-1.0, max=1.0)

            if return_img_per_step:
                imgs_per_step.append(inp_imgs.clone().detach())

        # Reactivate gradients for parameters for training
        for p in model.parameters():
            p.requires_grad = True
        model.train(is_training)

        # Reset gradient calculation to setting before this function
        torch.set_grad_enabled(had_gradients_enabled)

        if return_img_per_step:
            return torch.stack(imgs_per_step, dim=0)
        else:
            return inp_imgs


class Swish(nn.Module):

    def forward(self, x):
        return x * torch.sigmoid(x)


class CNNModel(nn.Module):

    def __init__(self, hidden_features=32, out_dim=1, **kwargs):
        super().__init__()
        # We increase the hidden dimension over layers. Here pre-calculated for simplicity.
        c_hid1 = hidden_features // 2
        c_hid2 = hidden_features
        c_hid3 = hidden_features * 2

        # Series of convolutions and Swish activation functions
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(
                1, c_hid1, kernel_size=5, stride=2, padding=4
            ),  # [16x16] - Larger padding to get 32x32 image
            Swish(),
            nn.Conv2d(c_hid1, c_hid2, kernel_size=3, stride=2, padding=1),  #  [8x8]
            Swish(),
            nn.Conv2d(c_hid2, c_hid3, kernel_size=3, stride=2, padding=1),  # [4x4]
            Swish(),
            nn.Conv2d(c_hid3, c_hid3, kernel_size=3, stride=2, padding=1),  # [2x2]
            Swish(),
            nn.Flatten(),
            nn.Linear(c_hid3 * 4, c_hid3),
            Swish(),
            nn.Linear(c_hid3, out_dim),
        )

    def forward(self, x):
        x = self.cnn_layers(x).squeeze(dim=-1)
        return x


class EBM(pl.LightningModule):
    def __init__(
        self,
        batch_size,
        img_shape=(1, 28, 28),
        alpha=0.1,
        lr=1e-4,
        beta1=0.0,
        step_size=0.1,
        in_channel=3,
        num_steps=256,
    ):
        super().__init__()
        self.save_hyperparameters()
        # from torchvision.models import mobilenet_v3_large

        # self.cnn = mobilenet_v3_large(weights='IMAGENET1K_V1')
        from torchvision.models import resnet18, ResNet18_Weights
        import timm

        # self.cnn  = timm.create_model('convnext_tiny.in12k', pretrained=True, num_classes=1, in_chans=in_channel)
        # self.cnn = timm.create_model(
        #     "mobilenetv3_small_100",
        #     in_chans=in_channel,          # 单通道输入
        #     pretrained=False,    # 不使用预训练权重（通道数不匹配）
        #     num_classes=1       # 分类类别数（例如 MNIST 的 10 类）
        # )
        self.cnn = CNNModel()

        # self.cnn = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1, in_channel=in_channel, num_classes=1)

        self.sampler = Sampler(self.cnn, img_shape=img_shape, sample_size=batch_size)
        self.example_input_array = torch.zeros(1, *img_shape)
        self.step_size = step_size
        self.num_steps = num_steps

    def forward(self, x):
        z = self.cnn(x)
        return z  # z is the negative of the energy

    def configure_optimizers(self):
        # Energy models can have issues with momentum as the loss surfaces changes with its parameters.
        # Hence, we set it to 0 by default.
        optimizer = optim.AdamW(
            self.parameters(), betas=(0.5, 0.999), lr=self.hparams.lr
        )
        # scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.97) # Exponential decay over epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "loss/average",
        }

    def training_step(self, batch, batch_idx):
        # We add minimal noise to the original images to prevent the model from focusing on purely "clean" inputs
        real_imgs = batch
        small_noise = torch.randn_like(real_imgs) * 0.005
        real_imgs.add_(small_noise).clamp_(min=-1.0, max=1.0)

        # Obtain samples
        fake_imgs = self.sampler.sample_new_exmps(steps=60, step_size=10, device=real_imgs.device)
        # steps=self.num_steps, step_size=self.step_size, device=real_imgs.device
        # )

        # Predict energy score for all images
        inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0)
        real_out, fake_out = self.cnn(inp_imgs).chunk(2, dim=0)

        # Calculate losses
        reg_loss = self.hparams.alpha * (real_out**2 + fake_out**2).mean()
        cdiv_loss = fake_out.mean() - real_out.mean()
        loss = reg_loss + cdiv_loss

        # Logging
        self.log("loss/average", loss, sync_dist=True)
        self.log("loss/reg", reg_loss, sync_dist=True)
        self.log("loss/cd", cdiv_loss, sync_dist=True)

        self.log("energy/real_mean", -real_out.mean(), sync_dist=True)
        self.log("energy/real_std", -real_out.std(), sync_dist=True)
        self.log("energy/fake_mean", -fake_out.mean(), sync_dist=True)
        self.log("energy/fake_std", -fake_out.std(), sync_dist=True)
        # log training images
        if self.trainer.global_rank == 0 and self.global_step % 200 == 0:
            self.logger.experiment.add_image(
                "real_imgs",
                torchvision.utils.make_grid(real_imgs[:16], nrow=4, normalize=True),
                global_step=self.global_step,
            )
        # norms = [p.grad.norm() for p in self.parameters() if p.grad is not None]
        # grad_norm = torch.norm(torch.stack(norms))
        # self.log('grad_norm', grad_norm, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # For validating, we calculate the contrastive divergence between purely random images and unseen examples
        # Note that the validation/test step of energy-based models depends on what we are interested in the model
        real_imgs = batch
        fake_imgs = torch.rand_like(real_imgs) * 2 - 1

        inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0)
        real_out, fake_out = self.cnn(inp_imgs).chunk(2, dim=0)

        cdiv = fake_out.mean() - real_out.mean()
        self.log("loss/val_cd", cdiv, sync_dist=True)
        self.log("loss/val_fake_energy", fake_out.mean(), sync_dist=True)
        self.log("loss/val_real_energy", real_out.mean(), sync_dist=True)


class GenerateCallback(pl.Callback):
    """is used for adding image generations to the model during training. After every  epochs (usually
    to reduce output to TensorBoard), we take a small batch of random images and perform
    many MCMC iterations until the model’s generation converges. Compared to the training that used
    60 iterations, we use 256 here because (1) we only have to do it once compared to the training
    that has to do it every iteration, and (2) we do not start from a buffer here, but from scratch.
    """

    def __init__(
        self, batch_size=8, vis_steps=8, num_steps=256, every_n_epochs=5, step_size=0.1
    ):
        super().__init__()
        self.batch_size = batch_size  # Number of images to generate
        self.vis_steps = vis_steps  # Number of steps within generation to visualize
        self.num_steps = num_steps  # Number of steps to take during generation
        self.every_n_epochs = every_n_epochs  # Only save those images every N epochs (otherwise tensorboard gets quite large)
        self.step_size = step_size

    def on_train_epoch_end(self, trainer, pl_module):
        # Skip for all other epoch
        if (
            trainer.current_epoch % self.every_n_epochs == 0
            and trainer.global_rank == 0
        ):
            # Generate images
            imgs_per_step = self.generate_imgs(pl_module)
            # shape [step, batchsize, 1, 28, 28]
            # Plot and add to tensorboard
            images = []
            for i in range(imgs_per_step.shape[1]):
                step_size = self.num_steps // self.vis_steps
                imgs_to_plot = imgs_per_step[
                    step_size - 1 :: step_size, i
                ]  # [vis_steps, 1, 28, 28]
                images.append(imgs_to_plot)
            # images shape [batch_size, vis_steps, 1, 28, 28]
            grid = torchvision.utils.make_grid(
                torch.cat(images, dim=0), nrow=len(images), normalize=True
            )
            trainer.logger.experiment.add_image(
                f"generation", grid, global_step=trainer.current_epoch
            )

    def on_validation_epoch_end(self, trainer, pl_module):
        return self.on_train_epoch_end(trainer, pl_module)

    def generate_imgs(self, pl_module):
        pl_module.eval()
        start_imgs = torch.rand((self.batch_size,) + pl_module.hparams["img_shape"]).to(
            pl_module.device
        )
        start_imgs = start_imgs * 2 - 1
        torch.set_grad_enabled(True)  # Tracking gradients for sampling necessary
        imgs_per_step = Sampler.generate_samples(
            pl_module.cnn,
            start_imgs,
            steps=self.num_steps,
            step_size=self.step_size,
            return_img_per_step=True,
        )
        torch.set_grad_enabled(False)
        pl_module.train()
        return imgs_per_step


class SamplerCallback(pl.Callback):
    """
    simply adds a randomly picked subset of images in the sampling buffer to the TensorBoard.
    This helps to understand what images are currently shown to the model as “fake”.
    """

    def __init__(self, num_imgs=32, every_n_epochs=5):
        super().__init__()
        self.num_imgs = num_imgs  # Number of images to plot
        self.every_n_epochs = every_n_epochs  # Only save those images every N epochs (otherwise tensorboard gets quite large)

    def on_train_epoch_end(self, trainer, pl_module):
        if (
            trainer.current_epoch % self.every_n_epochs == 0
            and trainer.global_rank == 0
        ):
            exmp_imgs = torch.cat(
                random.choices(pl_module.sampler.examples, k=self.num_imgs), dim=0
            )
            grid = torchvision.utils.make_grid(exmp_imgs, nrow=4, normalize=True)
            trainer.logger.experiment.add_image(
                "sampler", grid, global_step=trainer.current_epoch
            )


class OutlierCallback(pl.Callback):
    """
    This callback evaluates the model by recording the (negative) energy assigned to random noise.
    While our training loss is almost constant across iterations, this score is likely showing the
    progress of the model to detect “outliers”.
    """

    def __init__(self, batch_size=1024):
        super().__init__()
        self.batch_size = batch_size

    def on_train_epoch_end(self, trainer, pl_module):
        with torch.no_grad():
            pl_module.eval()
            rand_imgs = torch.rand(
                (self.batch_size,) + pl_module.hparams["img_shape"]
            ).to(pl_module.device)
            rand_imgs = rand_imgs * 2 - 1.0
            rand_out = pl_module.cnn(rand_imgs).mean()
            pl_module.train()

        trainer.logger.experiment.add_scalar(
            "rand_out", rand_out, global_step=trainer.current_epoch
        )


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        from torchvision.datasets import MNIST

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        train_set = MNIST(
            root="~/.cache/MNIST", train=True, transform=transform, download=True
        )
        test_set = MNIST(
            root="~/.cache/MNIST", train=False, transform=transform, download=True
        )

        class WrapperDataset(Dataset):
            def __init__(self, train_set):
                self.dataset = train_set

            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, idx):
                image, _ = self.dataset[idx]
                return image

        self.train_dataset = WrapperDataset(train_set)
        self.val_dataset = WrapperDataset(test_set)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
        )


class AnimeDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64, val_ratio=0.1):
        super().__init__()
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def setup(self, stage=None):
        # 加载完整训练集
        from torch.utils.data import random_split

        full_dataset = load_dataset("jlbaker361/anime_faces_dim_128_40k", split="train")

        # 转换为PyTorch Dataset
        class WrapperDataset(Dataset):
            def __init__(self, hf_dataset, transform):
                self.dataset = hf_dataset
                self.transform = transform

            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, idx):
                image = self.dataset[idx]["image"]
                return self.transform(image)

        full_torch_dataset = WrapperDataset(full_dataset, self.transform)

        # 手动划分训练集和验证集
        val_size = int(len(full_torch_dataset) * self.val_ratio)
        train_size = len(full_torch_dataset) - val_size

        self.train_dataset, self.val_dataset = random_split(
            full_torch_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),  # 固定随机种子保证可重复性
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=4, pin_memory=True
        )


class DebugCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("✅ 训练开始回调触发！")


def main(config):
    # Set random seed for reproducibility
    pl.seed_everything(42)

    # Dataset specific configurations
    dataset_configs = {
        "mnist": {
            "img_shape": (1, 28, 28),
            "data_module": MNISTDataModule,
            "in_channel": 1,
        },
        "anime": {
            "img_shape": (3, 128, 128),
            "data_module": AnimeDataModule,
            "in_channel": 3,
        },
    }

    # Get dataset specific configuration
    dataset_name = config.get("dataset", "mnist")
    dataset_config = dataset_configs[dataset_name]

    # Create data module
    data_module = dataset_config["data_module"](batch_size=config.get("batch_size", 64))

    # Create model
    batch_size = config.get("batch_size", 64)
    model = EBM(
        batch_size=batch_size,
        img_shape=dataset_config["img_shape"],
        alpha=config.get("alpha", 0.1),
        lr=config.get("lr", 1e-4),
        beta1=config.get("beta1", 0.0),
        step_size=config.get("step_size", 0.1),
        in_channel=dataset_config["in_channel"],
        num_steps=config.get("num_steps", 256),
    )

    # Create callbacks
    callbacks = [
        ModelCheckpoint(monitor="loss/val_cd", mode="min"),
        GenerateCallback(
            batch_size=7,
            vis_steps=8,
            num_steps=config["num_steps"],
            every_n_epochs=5,
            step_size=config.get("step_size", 0.1),
        ),
        SamplerCallback(num_imgs=32, every_n_epochs=1),
        OutlierCallback(batch_size=1024),
        LearningRateMonitor("epoch"),
    ]

    # Create trainer
    log_dir = config.get("log_dir")
    trainer = pl.Trainer(
        default_root_dir=f"{log_dir}/{dataset_name}_ebm",
        accelerator="auto",
        devices=1,
        max_epochs=config.get("max_epochs", 200),
        callbacks=callbacks,
        logger=TensorBoardLogger(log_dir, name=f"{dataset_name}_ebm"),
        precision="32",
        strategy="ddp",
        gradient_clip_val=0.1,
    )

    # Train model
    trainer.fit(model, data_module)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    torch.backends.cudnn.benchmark = True

    # Configuration
    config = {
        "dataset": "mnist",  # 'mnist' or 'anime'
        "batch_size": 256,
        "max_epochs": 200,
        "lr": 1e-4,
        "alpha": 0.1,
        "beta1": 0.0,
        "num_steps": 256,
        "step_size": 10,
        "log_dir": "/mnt/nas2/tdd/data-sync/minio/face/lilong/logs/ebm_anime/",
    }

    main(config)
