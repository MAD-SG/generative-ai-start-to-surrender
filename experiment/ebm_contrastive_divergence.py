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
import numpy as np
from torch import optim
import random

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
        self.examples = [(torch.rand((1,)+img_shape)*2-1) for _ in range(self.sample_size)]

    def sample_new_exmps(self, steps=60, step_size=10, device=None):
        """
        Function for getting a new batch of "fake" images.
        Inputs:
            steps - Number of iterations in the MCMC algorithm
            step_size - Learning rate nu in the algorithm above
        """
        # Choose 95% of the batch from the buffer, 5% generate from scratch
        n_new = np.random.binomial(self.sample_size, 0.05)
        rand_imgs = torch.rand((n_new,) + self.img_shape) * 2 - 1
        old_imgs = torch.cat(random.choices(self.examples, k=self.sample_size-n_new), dim=0)
        inp_imgs = torch.cat([rand_imgs, old_imgs], dim=0).detach().to(device)

        # Perform MCMC sampling
        inp_imgs = Sampler.generate_samples(self.model, inp_imgs, steps=steps, step_size=step_size)

        # Add new images to the buffer and remove old ones if needed
        self.examples = list(inp_imgs.to(torch.device("cpu")).chunk(self.sample_size, dim=0)) + self.examples
        self.examples = self.examples[:self.max_len]
        return inp_imgs

    @staticmethod
    def generate_samples(model, inp_imgs, steps=60, step_size=10, return_img_per_step=False):
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


class EBM(pl.LightningModule):
    def __init__(self, batch_size,img_shape=(3,128,128), alpha=0.1, lr=1e-4, beta1=0.0):
        super().__init__()
        self.save_hyperparameters()
        from torchvision.models import mobilenet_v3_large

        self.cnn = mobilenet_v3_large(weights='IMAGENET1K_V1')
        # 修改后的分类头
        self.cnn.classifier = nn.Sequential(
            nn.Linear(960, 512),  # 原始输入特征维度960
            nn.Hardswish(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

        self.sampler = Sampler(self.cnn, img_shape=img_shape, sample_size=batch_size)
        self.example_input_array = torch.zeros(1, *img_shape)

    def forward(self, x):
        z = self.cnn(x)
        return z

    def configure_optimizers(self):
        # Energy models can have issues with momentum as the loss surfaces changes with its parameters.
        # Hence, we set it to 0 by default.
        params = [
            {'params': self.cnn.features.parameters(), 'lr': self.hparams.lr*0.1},
            {'params': self.cnn.classifier.parameters(), 'lr': self.hparams.lr}
        ]
        optimizer = optim.AdamW(params, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.97) # Exponential decay over epochs
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # We add minimal noise to the original images to prevent the model from focusing on purely "clean" inputs
        real_imgs = batch
        small_noise = torch.randn_like(real_imgs) * 0.005
        real_imgs.add_(small_noise).clamp_(min=-1.0, max=1.0)

        # Obtain samples
        fake_imgs = self.sampler.sample_new_exmps(steps=60, step_size=10, device=real_imgs.device)

        # Predict energy score for all images
        inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0)
        real_out, fake_out = self.cnn(inp_imgs).chunk(2, dim=0)

        # Calculate losses
        reg_loss = self.hparams.alpha * (real_out ** 2 + fake_out ** 2).mean()
        cdiv_loss = fake_out.mean() - real_out.mean()
        loss = reg_loss + cdiv_loss

        # Logging
        self.log('loss', loss)
        self.log('loss_regularization', reg_loss)
        self.log('loss_contrastive_divergence', cdiv_loss)
        self.log('metrics_avg_real', real_out.mean())
        self.log('metrics_avg_fake', fake_out.mean())
        return loss


    def validation_step(self, batch, batch_idx):
        # For validating, we calculate the contrastive divergence between purely random images and unseen examples
        # Note that the validation/test step of energy-based models depends on what we are interested in the model
        real_imgs= batch
        fake_imgs = torch.rand_like(real_imgs) * 2 - 1

        inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0)
        real_out, fake_out = self.cnn(inp_imgs).chunk(2, dim=0)

        cdiv = fake_out.mean() - real_out.mean()
        self.log('val_contrastive_divergence', cdiv)
        self.log('val_fake_out', fake_out.mean())
        self.log('val_real_out', real_out.mean())

class GenerateCallback(pl.Callback):
    """is used for adding image generations to the model during training. After every  epochs (usually
    to reduce output to TensorBoard), we take a small batch of random images and perform
    many MCMC iterations until the model’s generation converges. Compared to the training that used
    60 iterations, we use 256 here because (1) we only have to do it once compared to the training
    that has to do it every iteration, and (2) we do not start from a buffer here, but from scratch.
    """

    def __init__(self, batch_size=8, vis_steps=8, num_steps=256, every_n_epochs=5):
        super().__init__()
        self.batch_size = batch_size         # Number of images to generate
        self.vis_steps = vis_steps           # Number of steps within generation to visualize
        self.num_steps = num_steps           # Number of steps to take during generation
        self.every_n_epochs = every_n_epochs # Only save those images every N epochs (otherwise tensorboard gets quite large)

    def on_epoch_end(self, trainer, pl_module):
        # Skip for all other epochs
        if (trainer.current_epoch +1) % self.every_n_epochs == 0:
            # Generate images
            imgs_per_step = self.generate_imgs(pl_module)
            # Plot and add to tensorboard
            for i in range(imgs_per_step.shape[1]):
                step_size = self.num_steps // self.vis_steps
                imgs_to_plot = imgs_per_step[step_size-1::step_size,i]
                grid = torchvision.utils.make_grid(imgs_to_plot, nrow=imgs_to_plot.shape[0], normalize=True, range=(-1,1))
                trainer.logger.experiment.add_image(f"generation_{i}", grid, global_step=trainer.current_epoch)

    def generate_imgs(self, pl_module):
        pl_module.eval()
        start_imgs = torch.rand((self.batch_size,) + pl_module.hparams["img_shape"]).to(pl_module.device)
        start_imgs = start_imgs * 2 - 1
        torch.set_grad_enabled(True)  # Tracking gradients for sampling necessary
        imgs_per_step = Sampler.generate_samples(pl_module.cnn, start_imgs, steps=self.num_steps, step_size=10, return_img_per_step=True)
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
        self.num_imgs = num_imgs             # Number of images to plot
        self.every_n_epochs = every_n_epochs # Only save those images every N epochs (otherwise tensorboard gets quite large)

    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            exmp_imgs = torch.cat(random.choices(pl_module.sampler.examples, k=self.num_imgs), dim=0)
            grid = torchvision.utils.make_grid(exmp_imgs, nrow=4, normalize=True, range=(-1,1))
            trainer.logger.experiment.add_image("sampler", grid, global_step=trainer.current_epoch)

class OutlierCallback(pl.Callback):
    """
    This callback evaluates the model by recording the (negative) energy assigned to random noise.
    While our training loss is almost constant across iterations, this score is likely showing the
    progress of the model to detect “outliers”.
    """

    def __init__(self, batch_size=1024):
        super().__init__()
        self.batch_size = batch_size

    def on_epoch_end(self, trainer, pl_module):
        with torch.no_grad():
            pl_module.eval()
            rand_imgs = torch.rand((self.batch_size,) + pl_module.hparams["img_shape"]).to(pl_module.device)
            rand_imgs = rand_imgs * 2 - 1.0
            rand_out = pl_module.cnn(rand_imgs).mean()
            pl_module.train()

        trainer.logger.experiment.add_scalar("rand_out", rand_out, global_step=trainer.current_epoch)


class AnimeDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64, val_ratio=0.1):
        super().__init__()
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

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
            generator=torch.Generator().manual_seed(42)  # 固定随机种子保证可重复性
        )
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True
        )

def main(config):
    # Initialize model and data
    model = EBM(
        config['batch_size'],img_shape=config['img_shape'],alpha=config['alpha'],
        lr=config['lr'], beta1=config['beta1'],
        )
    data_module = AnimeDataModule(batch_size=config['batch_size'], val_ratio=config['val_ratio'])

    # Setup logger and callbacks
    logger = TensorBoardLogger(config['log_dir'], name="ebm_anime")
    checkpoint_callback = ModelCheckpoint(
        monitor='loss',
        dirpath='checkpoints',
        filename='ebm-anime-{epoch:02d}-{loss:.2f}',
        save_top_k=3,
        mode='min',
    )

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config['epochs'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=-1,
        precision='16-mixed',
        logger=logger,
        callbacks=[
            checkpoint_callback,
            GenerateCallback(every_n_epochs=5),
            SamplerCallback(every_n_epochs=5),
            OutlierCallback(batch_size=config['batch_size']),
            LearningRateMonitor("epoch")
        ],
        log_every_n_steps=10,
        gradient_clip_val=1.0,
    )

    # Train model
    trainer.fit(model, data_module)

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')  # Balanced between performance and precision
    torch.backends.cudnn.benchmark = True

    config = {
        'batch_size': 64,
        'val_ratio': 0.02,
        'img_shape': (3,128,128),
        'alpha': 0.1,
        'lr': 1e-4,
        'beta1': 0.0,
        'epochs': 100,
        'log_dir': '/mnt/nas2/tdd/data-sync/minio/face/lilong/logs/ebm_anime',
    }
    main(config)