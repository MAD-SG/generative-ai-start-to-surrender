import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import math
from torchvision import transforms
from datasets import load_dataset
import torchvision
from diffusers.models import UNet2DModel

class ScoreNetwork(nn.Module):
    """Score network using diffusers' UNet2D."""
    def __init__(self, channels: int = 3, base_channels: int = 128):
        super().__init__()

        # UNet from diffusers
        self.unet = UNet2DModel(
            in_channels=channels,  # RGB images
            out_channels=channels,  # Score prediction for each channel
            sample_size=128,  # Image size
            layers_per_block=1,  # Number of resnet blocks per level
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
            flip_sin_to_cos=True,
            norm_num_groups = 16,
            # use_timestep_embedding=True,
        )

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
        beta2: float = 0.999
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

        # Create score network
        self.score_net = ScoreNetwork(channels=channels, base_channels=base_channels)

        # Calculate sigma levels (noise scales)
        self.sigmas = torch.exp(torch.linspace(
            np.log(sigma_min), np.log(sigma_max), num_scales
        )).cuda()

        self.sigmas = torch.Tensor(np.linspace(sigma_min,sigma_max,num_scales)).cuda()

        print('self.sigmas', self.sigmas)

        # Initialize validation step counter
        self.val_step_count = 0

    def forward(self, x: torch.Tensor, sigma_idx: int) -> torch.Tensor:
        # Convert sigma index to timestep for the UNet
        # Scale to [0, 999] range which is commonly used in diffusion models
        if not isinstance(sigma_idx, torch.Tensor):
            sigma_idx = torch.tensor(sigma_idx, device=x.device)
        t = (sigma_idx / self.num_scales * 999).expand(x.shape[0]).long().to(x.device)
        return self.score_net(x, t)

    def compute_loss(self, x: torch.Tensor, batch_size: Optional[int] = None) -> torch.Tensor:
        batch_size = x.shape[0]

        # Randomly select sigma indices for each sample in the batch
        sigma_indices = torch.randint(0, self.num_scales, (batch_size,), device=x.device)
        self.sigmas = self.sigmas.to(x.device)
        sigmas = self.sigmas[sigma_indices]

        # Expand sigmas to match input dimensions
        sigmas = sigmas.view(-1, 1, 1, 1)

        # Generate noise and perturb input
        noise = torch.randn_like(x) * sigmas
        perturbed_x = x + noise

        # Compute score
        score = self.forward(perturbed_x, sigma_indices)
        target = -noise / (sigmas ** 2)

        # Compute loss
        loss = 0.5 * (sigmas ** 2) * F.mse_loss(score, target, reduction='none')

        # Log losses for specific sigma levels
        # with torch.no_grad():
        #     if self.training:
        #         # Log min sigma loss
        #         min_mask = (sigma_indices == 0)
        #         if min_mask.any():
        #             min_loss = loss[min_mask].detach().mean()
        #             self.log('loss_sigma_min', min_loss, prog_bar=True, sync_dist=True)

        #         # Log medium sigma loss
        #         med_mask = (sigma_indices == self.num_scales // 2)
        #         if med_mask.any():
        #             med_loss = loss[med_mask].detach().mean()
        #             self.log('loss_sigma_med', med_loss, prog_bar=True, sync_dist=True)

        #         # Log max sigma loss
        #         max_mask = (sigma_indices == self.num_scales - 1)
        #         if max_mask.any():
        #             max_loss = loss[max_mask].detach().mean()
        #             self.log('loss_sigma_max', max_loss, prog_bar=True, sync_dist=True)

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
        if batch_idx == 0 and self.val_step_count % 5 == 0:
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
        sigma_weights = self.sigmas / self.sigmas.sum()  # Normalized weights for each sigma
        steps_per_sigma = (num_steps * sigma_weights).int()  # Allocate steps proportionally

        for sigma_idx in reversed(range(self.num_scales)):
            sigma = self.sigmas[sigma_idx]
            step_size = eps * (sigma ** 2)

            for _ in range(steps_per_sigma[sigma_idx]):
                noise = torch.randn_like(x) * (2 * step_size)
                score = self.forward(x, torch.tensor(sigma_idx, device=x.device))
                x = x + step_size * score + noise

        return x

    def _log_training_batch(self, batch: torch.Tensor):
        """Log original and noised versions of training batch."""
        if not isinstance(self.logger, pl.loggers.TensorBoardLogger):
            return

        with torch.no_grad():
            # Original images
            grid_clean = torchvision.utils.make_grid(
                batch[:16].clamp(-1, 1),
                nrow=4,
                normalize=True,
                value_range=(-1, 1)
            )

            # Noised versions at different scales
            noised_grids = []
            for sigma_idx in [0, len(self.sigmas)//2, -1]:  # Start, middle, end noise levels
                sigma = self.sigmas[sigma_idx]
                noised_batch = batch[:16] + torch.randn_like(batch[:16]) * sigma
                grid_noised = torchvision.utils.make_grid(
                    noised_batch.clamp(-1, 1),
                    nrow=4,
                    normalize=True,
                    value_range=(-1, 1)
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

    def _log_generated_samples(self, num_samples: int = 16):
        """Generate and log samples to TensorBoard."""
        if not isinstance(self.logger, pl.loggers.TensorBoardLogger):
            return

        device = next(self.parameters()).device
        with torch.no_grad():
            # Generate samples
            samples = self.sample(num_samples, device)

            # Create and log image grid
            grid = torchvision.utils.make_grid(
                samples.clamp(-1, 1),
                nrow=4,
                normalize=True,
                value_range=(-1, 1)
            )

            # Log to TensorBoard
            self.logger.experiment.add_image(
                'generated/samples',
                grid,
                self.global_step
            )

            # Log intermediate steps of generation
            # if self.current_epoch % 10 == 0:  # Less frequent to save space
            #     x = torch.randn(num_samples, 3, 128, 128).to(device)
            #     for sigma_idx in reversed(range(0, self.num_scales, self.num_scales//4)):
            #         sigma = self.sigmas[sigma_idx]
            #         step_size = 2e-5 * (sigma ** 2)

            #         # Single Langevin step
            #         noise = torch.randn_like(x) *(2 * step_size)
            #         score = self.forward(x, sigma_idx)
            #         x = x + step_size * score + noise

            #         # Log intermediate result
            #         grid = torchvision.utils.make_grid(
            #             x.clamp(-1, 1),
            #             nrow=4,
            #             normalize=True,
            #             value_range=(-1, 1)
            #         )
            #         self.logger.experiment.add_image(
            #             f'generated/intermediate_scale_{sigma_idx}',
            #             grid,
            #             self.global_step
            #         )

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
    data_module = AnimeDataModule(batch_size=64)

    # Initialize model
    model = MultiScaleDSM(
        channels=3,
        base_channels=64,
        sigma_min=0.095,
        sigma_max=1.2,
        num_scales=10,
        lr=2e-4
    )

    # Initialize trainer with TensorBoard logger
    trainer = pl.Trainer(
        default_root_dir="/mnt/nas2/tdd/data-sync/minio/face/lilong/logs/ebm_anime/",
        max_epochs=100,
        accelerator='auto',
        devices=-1,
        precision='bf16',
        gradient_clip_val=0.1,
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
                save_top_k=3,
                mode='min',
                auto_insert_metric_name=False


            ),
            # Add learning rate monitoring
            pl.callbacks.LearningRateMonitor(logging_interval='step')
        ]
    )

    # Train model
    trainer.fit(model, data_module,  ckpt_path='/mnt/nas2/tdd/data-sync/minio/face/lilong/logs/ebm_anime/dsm_anime/version_21/checkpoints/dsm_22_0.12.ckpt')

if __name__ == "__main__":
    torch.set_float32_matmul_precision('medium')
    main()