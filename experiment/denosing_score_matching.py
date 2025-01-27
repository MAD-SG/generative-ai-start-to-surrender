import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from typing import List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import math
from torchvision import transforms
from datasets import load_dataset
import torchvision
from diffusers.models import UNet2DModel

def logit_transform(image, lam=1e-6):
    """input is a normalized image in range [0,1], this function map it to (-inf,inf), where the inverse function is the sigmoid function"""
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)

def add_noise_input(image):
    """used to handle the low density areas"""
    image = image/256. * 255 + torch.rand_like(image) / 256.
    return image

class ScoreNetwork(nn.Module):
    """Score network using diffusers' UNet2D."""
    def __init__(self, channels: int = 3, base_channels: int = 128):
        super().__init__()

        # UNet from diffusers
        # self.unet = UNet2DModel(
        #     in_channels=channels,  # RGB images
        #     out_channels=channels,  # Score prediction for each channel
        #     sample_size=128,  # Image size
        #     layers_per_block=2,  # Number of resnet blocks per level
        #     block_out_channels=(base_channels, base_channels*2, base_channels*4, base_channels*8),
        #     time_embedding_type="fourier"  # Sinusoidal embedding
        # )

        self.unet = UNet2DModel(
            in_channels=channels,  # RGB images
            out_channels=channels,  # Score prediction for each channel
            sample_size=128,  # Image size
            layers_per_block=2,  # Number of resnet blocks per level
            block_out_channels=(base_channels, base_channels*2, base_channels*4, base_channels*8),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
            # Enable time conditioning
            time_embedding_type="positional",
            # time_embedding_type="fourier",
            flip_sin_to_cos=True,
            norm_num_groups = 16,
            # use_timestep_embedding=True,
        )
        # model = UNet2DModel.from_pretrained("google/ddpm-cifar10-32")

        # Modify the model's configuration
        # model.config.sample_size = 128
        # self.unet = model

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # The UNet expects timesteps as float
        t = t.float()
        return self.unet(x, t).sample

class MultiScaleDSM(pl.LightningModule):
    """PyTorch Lightning module for Multi-scale Denoising Score Matching."""
    def __init__(
        self,
        channels: int = 3,
        base_channels: int = 32,
        sigma_min: float = 0.01,
        sigma_max: float = 0.5,
        num_scales: int = 10,
        lr: float = 2e-4,
        beta1: float = 0.5,
        beta2: float = 0.999,
        curriculum_epochs: int = 100  # Number of epochs to complete the curriculum
    ):
        super().__init__()
        self.save_hyperparameters()

        # Model parameters
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.num_scales = num_scales
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.curriculum_epochs = curriculum_epochs

        # Create score network
        self.score_net = ScoreNetwork(channels=channels, base_channels=base_channels)

        # Calculate sigma levels (noise scales)
        sigmas = torch.exp(torch.linspace(
            np.log(sigma_min), np.log(sigma_max), num_scales
        ))
        sigmas = torch.round(sigmas, decimals=3)
        self.register_buffer('sigmas', sigmas)
        time_steps = self.sigma2time_step(self.sigmas)
        print('self.sigmas', self.sigmas)
        print("t used in unet for conditional noise prediction", time_steps)

        # Initialize validation step counter
        self.val_step_count = 0

    def sigma2time_step(self, sigma):
        sigma = (sigma - self.sigma_min) / (self.sigma_max - self.sigma_min)
        sigma = sigma*999
        return sigma.long()

    def forward(self, x: torch.Tensor, sigma: Union[float, torch.Tensor]) -> torch.Tensor:
        # Convert sigma index to timestep for the UNet
        # Scale to [0, 999] range which is commonly used in diffusion models
        if not isinstance(sigma, torch.Tensor):
            sigma = torch.tensor(sigma, device=x.device)
        t = self.sigma2time_step(sigma)
        if t.numel()==1:
            t = t.view(-1).expand(x.shape[0])
        return self.score_net(x, t)

    def get_sigma_weights(self) -> torch.Tensor:
        """
        Calculate sampling weights for sigma indices based on training progress.
        - Start: Focus on small sigmas.
        - Middle: Gradually shift focus to large sigmas.
        - Final stage: Balance weights between small and large sigmas.
        """
        current_epoch = self.current_epoch
        progress = min(current_epoch / self.curriculum_epochs, 1.0)

        # Initialize weights with equal probabilities
        weights = torch.ones(self.num_scales, device=self.device)
        if not self.training:
            return weights / weights.sum()  # Equal probability during validation

        # Stage 1: Strong focus on small sigmas
        if progress < 0.33:  # First 33% of training
            decay = 2.0 * (0.33 - progress)  # Higher decay at start
            weights = torch.exp(-torch.arange(self.num_scales, device=self.device) * decay)

        # Stage 2: Gradual shift to larger sigmas
        elif progress < 0.66:  # Middle 33%-66% of training
            decay = 2.0 * (0.66 - progress)  # Decay decreases over time
            growth = 2.0 * (progress - 0.33)  # Growth increases over time
            weights = torch.exp(-torch.arange(self.num_scales, device=self.device) * decay) + \
                    torch.exp(torch.arange(self.num_scales, device=self.device) * growth)

        # Stage 3: Balance between small and large sigmas
        else:  # Final 33% of training
            balance = 1.0 + (progress - 0.66)  # Increase balance over time
            small_sigma_focus = torch.exp(-torch.arange(self.num_scales, device=self.device) * balance)
            large_sigma_focus = torch.exp(torch.arange(self.num_scales, device=self.device) * balance)
            weights = small_sigma_focus + large_sigma_focus

        # Normalize weights to probabilities
        weights = weights / weights.sum()
        return weights


    def compute_loss(self, x: torch.Tensor, batch_size: Optional[int] = None) -> torch.Tensor:
        batch_size = x.shape[0]

        # Get sampling weights based on training progress
        weights = self.get_sigma_weights()

        # Log sigma probabilities during training
        if self.training and self.global_step % 100 == 0:  # Log every 100 steps
            for sigma, prob in zip(self.sigmas, weights):
                self.log(f'sigma_prob/sigma_{sigma:.3f}', prob.item(), prog_bar=False, sync_dist=True)

        # Sample sigma indices according to weights
        sigma_indices = torch.multinomial(weights, batch_size, replacement=True)
        self.sigmas = self.sigmas.to(x.device)
        sigmas = self.sigmas[sigma_indices]

        # Expand sigmas to match input dimensions
        sigmas = sigmas.view(-1, 1, 1, 1)

        x = add_noise_input(x)

        # Generate noise and perturb input
        noise = torch.randn_like(x) * sigmas # each sample has its own noise
        perturbed_x = x + noise

        # Compute score
        score = self.forward(perturbed_x, sigmas.view(-1))
        target = -noise / (sigmas ** 2)

        # Compute loss

        loss = 0.5 * (sigmas ** 2) * (score - target) ** 2

        # Log losses for specific sigma levels
        with torch.no_grad():
            if self.training:
                # Log min sigma loss
                min_mask = (sigma_indices == 0) | (sigma_indices == 1)
                if min_mask.any():
                    min_loss = loss[min_mask].detach().mean()
                    self.log('loss/sigma_min', min_loss, prog_bar=False, sync_dist=False)

                # Log medium sigma loss
                med_mask = (sigma_indices == self.num_scales // 2) | (sigma_indices == self.num_scales // 2 + 1)
                if med_mask.any():
                    med_loss = loss[med_mask].detach().mean()
                    self.log('loss/sigma_med', med_loss, prog_bar=False, sync_dist=False)

                # Log max sigma loss
                max_mask = (sigma_indices == self.num_scales - 1) | (sigma_indices == self.num_scales - 2)
                if max_mask.any():
                    max_loss = loss[max_mask].detach().mean()
                    self.log('loss/sigma_max', max_loss, prog_bar=False, sync_dist=False)

        loss = loss.mean()

        return loss

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss = self.compute_loss(batch)
        # Log training metrics
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)

        # Log training batch and noised versions periodically
        if batch_idx == 0 and self.current_epoch % 5 == 0:
            self._log_training_batch(batch)

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        loss = self.compute_loss(batch)
        # Log validation metrics
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)

        # Generate and log samples during validation
        # if batch_idx == 0 and self.val_step_count % 5 == 0:
        # print("🔴 Validating...", self.val_step_count, self.local_rank, batch_idx)
        if self.val_step_count % 100 == 0:
            self._log_generated_samples()
        self.val_step_count += 1

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            betas=(self.beta1, self.beta2)
        )
        return optimizer


    def sample(
        self,
        num_samples: int,
        device: torch.device,
        num_steps: int = 100,
        eps: float = 2e-5
    ) -> torch.Tensor:
        """Generate samples using annealed Langevin dynamics."""
        shape = (num_samples, 3, 128, 128)  # Assuming 128x128 RGB images
        x = torch.randn(*shape).to(device)
        images = []
        for i in range(self.num_scales):
            sigma = self.sigmas[self.num_scales - 1 - i]
            sigma_step_size = eps * (sigma / self.sigmas[-1]) ** 2
            for _ in range(num_steps):
                noise = torch.randn_like(x) * math.sqrt(2 * sigma_step_size)
                with torch.no_grad():
                    score = self.forward(x, sigma)
                x = x + sigma_step_size * score + noise
                x = torch.clamp(x, 0, 1)
            images.append(x)
        images = torch.stack(images, dim=0) # shape [scales, num_samples, 3, 128, 128]
        return images

    def _log_training_batch(self, batch: torch.Tensor):
        """Log original and noised versions of training batch."""
        with torch.no_grad():
            # Original images
            grid_clean = torchvision.utils.make_grid(
                batch[:16].clamp(0, 1),
                nrow=4,
                normalize=True,
                value_range=(0, 1)
            )

            # Noised versions at different scales
            noised_grids = []
            for sigma_idx in [0, len(self.sigmas)//2, -1]:  # Start, middle, end noise levels
                sigma = self.sigmas[sigma_idx]
                noised_batch = batch[:16] + torch.randn_like(batch[:16]) * sigma
                grid_noised = torchvision.utils.make_grid(
                    noised_batch.clamp(0, 1),
                    nrow=4,
                    normalize=True,
                    value_range=(0, 1)
                )
                noised_grids.append(grid_noised)

            # Log to TensorBoard
            self.logger.experiment.add_image(
                'training/original_images',
                grid_clean,
                self.current_epoch
            )
            for idx, grid in enumerate(noised_grids):
                self.logger.experiment.add_image(
                    f'training/noised_images_level_{idx}',
                    grid,
                    self.current_epoch
                )

    def _log_generated_samples(self, num_samples: int = 2):
        """Generate and log samples to TensorBoard."""
        if not isinstance(self.logger, pl.loggers.TensorBoardLogger):
            return

        device = next(self.parameters()).device
        with torch.no_grad():
            # Generate samples
            samples = self.sample(num_samples, device)
            n_samples = samples.shape[1]
            # Create and log image grid
            samples = samples.permute(1, 0, 2, 3, 4).reshape(-1, samples.shape[2], samples.shape[3], samples.shape[4])
            grid = torchvision.utils.make_grid(
                samples.clamp(0, 1),
                nrow=n_samples,
                normalize=True,
                value_range=(0, 1)
            )

            # Log to TensorBoard
            name = f'generated/global_{self.global_step}_rank_{self.local_rank}_val_step_{self.val_step_count}'
            self.logger.experiment.add_image(
                name,
                grid,
                self.global_step
            )


class AnimeDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64, val_ratio=0.1):
        super().__init__()
        self.batch_size = batch_size
        self.val_ratio = val_ratio
        self.transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomGrayscale(p=0.1),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                transforms.ToTensor(),
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def setup(self, stage=None):
        # Load full training set
        full_dataset = load_dataset("jlbaker361/anime_faces_dim_128_40k", split="train")

        # Convert to PyTorch Dataset
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

        # Split into train and validation sets
        val_size = int(len(full_torch_dataset) * self.val_ratio)
        train_size = len(full_torch_dataset) - val_size

        self.train_dataset, self.val_dataset = random_split(
            full_torch_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
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
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
        )

def main():
    # Create data module
    data_module = AnimeDataModule(batch_size=200)

    # Initialize model
    model = MultiScaleDSM(
        channels=3,
        base_channels=64,
        sigma_min=0.01,
        sigma_max=1.2,
        num_scales=20,
        lr=1e-4,
        curriculum_epochs=2000,
    )

    # Initialize trainer with TensorBoard logger
    trainer = pl.Trainer(
        default_root_dir="/mnt/nas2/tdd/data-sync/minio/face/lilong/logs/ebm_anime/",
        max_epochs=2000,
        accelerator='auto',
        benchmark=True,
        deterministic=False,
        devices=-1,
        precision='bf16',
        gradient_clip_val=0.01,
        logger=pl.loggers.TensorBoardLogger(
            save_dir="/mnt/nas2/tdd/data-sync/minio/face/lilong/logs/ebm_anime/",
            name='dsm_anime',
            version=None  # Auto-increment version
        ),
        log_every_n_steps=10,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                monitor='val_loss',
                filename='dsm_{epoch:02d}_{val_loss:.3f}',
                # save_top_k=20,
                every_n_epochs=10,
                mode='min',
                auto_insert_metric_name=False
            ),
            # Add learning rate monitoring
            pl.callbacks.LearningRateMonitor(logging_interval='step')
        ]
    )

    # Train model
    ckpt_path='/mnt/nas2/tdd/data-sync/minio/face/lilong/logs/ebm_anime/dsm_anime/version_28/checkpoints/dsm_85_0.010.ckpt'

    trainer.fit(model, data_module, ckpt_path=None)
    #  ckpt_path='/mnt/nas2/tdd/data-sync/minio/face/lilong/logs/ebm_anime/dsm_anime/version_21/checkpoints/dsm_22_0.12.ckpt')

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    main()
    # ckpt_path='/mnt/nas2/tdd/data-sync/minio/face/lilong/logs/ebm_anime/dsm_anime/version_28/checkpoints/dsm_85_0.010.ckpt'
    # load_and_sample(ckpt_path, n_samples=4, save_dir='samples', )