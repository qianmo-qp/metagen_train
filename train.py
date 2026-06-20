"""
Training loop for Conditional DiT on MNIST with W&B monitoring.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import subprocess
import logging

# Disable torch._dynamo to avoid ONNX import issues
os.environ['TORCH_DISABLE_DYNANMO'] = '1'

# Setup W&B
os.environ["WANDB_API_KEY"] = "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"
import wandb

from dit_model import ConditionalDiT
from diffusion import create_diffusion

# Setup logging
logging.basicConfig(level=logging.INFO)


def load_mnist_idx(filepath):
    """Load MNIST IDX format files."""
    with open(filepath, 'rb') as f:
        # Read magic number and dimensions
        magic = int.from_bytes(f.read(4), byteorder='big')
        
        if magic == 2049:  # Labels
            num_items = int.from_bytes(f.read(4), byteorder='big')
            data = np.frombuffer(f.read(num_items), dtype=np.uint8)
        elif magic == 2051:  # Images
            num_images = int.from_bytes(f.read(4), byteorder='big')
            rows = int.from_bytes(f.read(4), byteorder='big')
            cols = int.from_bytes(f.read(4), byteorder='big')
            data = np.frombuffer(f.read(num_images * rows * cols), dtype=np.uint8)
            data = data.reshape(num_images, rows, cols)
        else:
            raise ValueError(f"Unknown magic number: {magic}")
    
    return data


def load_mnist_data(data_dir='data/minst'):
    """Load MNIST dataset."""
    train_images = load_mnist_idx(os.path.join(data_dir, 'train-images.idx3-ubyte'))
    train_labels = load_mnist_idx(os.path.join(data_dir, 'train-labels.idx1-ubyte'))
    test_images = load_mnist_idx(os.path.join(data_dir, 't10k-images.idx3-ubyte'))
    test_labels = load_mnist_idx(os.path.join(data_dir, 't10k-labels.idx1-ubyte'))
    
    return train_images, train_labels, test_images, test_labels


def normalize_images(images):
    """Normalize images to [-1, 1] range."""
    # Convert to float and scale to [0, 1]
    images = images.astype(np.float32) / 255.0
    # Scale to [-1, 1]
    images = 2.0 * images - 1.0
    # Add channel dimension
    images = np.expand_dims(images, axis=1)
    return images


class Trainer:
    def __init__(
        self,
        model,
        diffusion,
        device,
        learning_rate=1e-4,
        num_epochs=10,
        batch_size=128,
        checkpoint_dir='checkpoints',
        log_interval=100,
        use_wandb=False,
        sample_interval=10,
    ):
        self.model = model.to(device)
        self.diffusion = diffusion
        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval
        self.use_wandb = use_wandb
        self.sample_interval = sample_interval
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Create checkpoint directory
        Path(self.checkpoint_dir).mkdir(exist_ok=True)
        
        self.step = 0
        self.losses = []
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Sample random timesteps
            t = torch.randint(0, self.diffusion.timesteps, (images.shape[0],), device=self.device)
            
            # Forward diffusion: add noise
            x_t, noise = self.diffusion.q_sample(images, t)
            
            # Predict noise
            self.optimizer.zero_grad()
            noise_pred = self.model(x_t, t, labels)
            
            # Compute loss
            loss = self.loss_fn(noise_pred, noise)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            epoch_loss += loss.item()
            self.losses.append(loss.item())
            self.step += 1
            
            if (batch_idx + 1) % self.log_interval == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                print(f"Step {self.step}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {avg_loss:.6f}")
                
                # Log to W&B
                if self.use_wandb:
                    wandb.log({
                        "loss": avg_loss,
                        "step": self.step,
                        "batch": batch_idx + 1,
                        "epoch": int(self.step / len(train_loader)) + 1,
                    }, step=self.step)
        
        return epoch_loss / len(train_loader)
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'losses': self.losses,
        }
        
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"Saved best model to {best_path}")
    
    def generate_samples(self, epoch):
        """Generate samples using DDIM and optionally log to W&B."""
        import matplotlib.pyplot as plt
        from PIL import Image
        import io
        
        self.logger.info(f"Generating samples at epoch {epoch}...")
        self.model.eval()
        
        # Generate 1 sample per class using DDIM (50 steps = ~6 seconds)
        class_labels = torch.arange(10, device=self.device)
        
        # Use DDIM for fast sampling
        samples = self.diffusion.sample_ddim(
            self.model,
            num_samples=10,
            num_classes=10,
            device=self.device,
            num_steps=100,     # DDIM: 100 steps (更好质量!)
            eta=0.0,           # Deterministic sampling
            class_labels=class_labels
        )
        
        # Denormalize
        samples = (samples + 1.0) / 2.0
        samples = torch.clamp(samples, 0.0, 1.0)
        
        # Create grid image
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        axes = axes.flatten()
        
        for idx in range(10):
            axes[idx].imshow(samples[idx, 0].cpu().numpy(), cmap='gray')
            axes[idx].set_title(f'Class {idx}')
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        # Save locally
        sample_path = os.path.join('outputs', f'generated_samples_epoch_{epoch}.png')
        Path('outputs').mkdir(exist_ok=True)
        plt.savefig(sample_path, dpi=100, bbox_inches='tight')
        self.logger.info(f"Saved samples to {sample_path}")
        
        # Log to W&B with correct step counter
        if self.use_wandb:
            # Convert to PIL Image for W&B
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            pil_image = Image.open(buf)
            
            # Use monotonically increasing step (current step, not epoch)
            wandb.log({
                f"samples_epoch_{epoch}": wandb.Image(pil_image),
                "epoch": epoch,
            }, step=self.step)  # ← 使用self.step而不是epoch!
            self.logger.info(f"Logged samples to W&B for epoch {epoch} at step {self.step}")
        
        plt.close()
        self.model.train()
    
    def train(self, train_loader):
        """Train for multiple epochs."""
        print(f"Training on {self.device}")
        print(f"Total epochs: {self.num_epochs}")
        
        for epoch in range(self.num_epochs):
            print(f"\n=== Epoch {epoch + 1}/{self.num_epochs} ===")
            avg_loss = self.train_epoch(train_loader)
            print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.6f}")
            
            # Save checkpoint
            self.save_checkpoint(epoch + 1)
            
            # Generate samples every sample_interval epochs
            if (epoch + 1) % self.sample_interval == 0:
                self.logger.info(f"Sampling at epoch {epoch + 1}")
                try:
                    self.generate_samples(epoch + 1)
                except Exception as e:
                    self.logger.error(f"Error during sampling: {e}")
            
            # Log epoch metrics to W&B
            if self.use_wandb:
                # 不要用step参数,W&B会自动使用最后一个step
                wandb.log({
                    "epoch_loss": avg_loss,
                    "epoch": epoch + 1,
                })  # 移除step参数!
        
        print("\nTraining complete!")
        return self.losses


def main():
    # Setup
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device = 'cuda'
        print(f"✅ CUDA is available")
        print(f"   Device: {torch.cuda.get_device_name(0)}")
        print(f"   Compute Capability: {torch.cuda.get_device_capability(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = 'cpu'
        print(f"⚠️  CUDA not detected, using CPU")
        print(f"   To use GPU, ensure NVIDIA drivers and CUDA are installed")
        print(f"   Run: nvidia-smi (to check GPU)")
    
    print(f"\nUsing device: {device}")
    
    # Initialize W&B
    print("\n📊 Initializing W&B...")
    wandb.init(
        project='minst',
        name='conditional_dit_mnist',
        config={
            'num_epochs': 100,
            'batch_size': 128,
            'learning_rate': 1e-4,
            'timesteps': 1000,
            'model_type': 'ConditionalDiT',
            'hidden_dim': 192,
            'num_layers': 6,
            'num_heads': 3,
        }
    )
    print("✅ W&B initialized")
    
    # Hyperparameters
    num_epochs = 400  # 增加到400获得更优的模型
    batch_size = 128
    learning_rate = 1e-4
    timesteps = 1000
    sample_interval = 20  # 每20个epoch采样一次 (而不是10,减少计算)
    
    # Load MNIST
    print("Loading MNIST dataset...")
    train_images, train_labels, test_images, test_labels = load_mnist_data()
    
    # Normalize
    train_images = normalize_images(train_images)
    test_images = normalize_images(test_images)
    
    print(f"Train images shape: {train_images.shape}, dtype: {train_images.dtype}")
    print(f"Train labels shape: {train_labels.shape}, dtype: {train_labels.dtype}")
    
    # Create dataset
    train_dataset = TensorDataset(
        torch.from_numpy(train_images).float(),
        torch.from_numpy(train_labels).long()
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Create model and diffusion
    print("Creating model...")
    model = ConditionalDiT(
        img_size=28,
        patch_size=4,
        in_channels=1,
        hidden_dim=192,
        num_heads=3,
        num_layers=6,
        time_dim=256,
        num_classes=10,
        mlp_ratio=4,
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    diffusion = create_diffusion(timesteps=timesteps, schedule_type='cosine')
    
    # Move diffusion buffers to device
    for attr_name in dir(diffusion):
        attr = getattr(diffusion, attr_name)
        if isinstance(attr, torch.Tensor):
            setattr(diffusion, attr_name, attr.to(device))
    
    # Train
    trainer = Trainer(
        model=model,
        diffusion=diffusion,
        device=device,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        batch_size=batch_size,
        log_interval=100,
        use_wandb=True,  # 启用W&B
        sample_interval=sample_interval,  # 每10个epoch采样
    )
    
    losses = trainer.train(train_loader)
    
    print(f"\nTraining complete! Final checkpoint saved.")
    print(f"Total steps: {trainer.step}")
    
    # 关闭W&B
    wandb.finish()
    print("\n📊 W&B run finished")


if __name__ == "__main__":
    main()
