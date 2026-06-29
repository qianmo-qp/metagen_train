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


def load_phase_data(data_dir='data/minst_phase', use_cache=True):
    """
    Load phase hologram data from cached .npz files or build cache if not exists.
    
    缓存结构:
    data_dir/
    ├── train/
    │   ├── digit_0/phase_*.npy
    │   ├── digit_1/phase_*.npy
    │   └── ...
    │   └── phase_train_cache.npz  ← 缓存文件
    └── test/
        └── ...
    
    首次运行时会自动调用 PhaseCacheBuilder 生成缓存
    """
    # Handle both relative and absolute paths
    if not os.path.isabs(data_dir) and not os.path.exists(data_dir):
        # Try server path if relative path doesn't exist
        server_path = f'/mnt/model_data/qp/metagen_train/{data_dir}'
        if os.path.exists(server_path):
            data_dir = server_path
            print(f"[INFO] Using server path: {data_dir}")
    
    # 导入缓存构建器
    from ds.phase_cache_builder import PhaseCacheBuilder
    
    train_phase = None
    train_labels = None
    test_phase = None
    test_labels = None
    
    # 尝试加载缓存
    for split in ['train', 'test']:
        split_dir = os.path.join(data_dir, split)
        cache_path = os.path.join(split_dir, f'phase_{split}_cache.npz')
        
        if use_cache and os.path.exists(cache_path):
            print(f"[INFO] Loading {split} from cache: {cache_path}")
            try:
                cache = np.load(cache_path)
                phase = cache['phase'].astype(np.float32)
                labels = cache['labels'].astype(np.int64)
                print(f"✓ {split.upper()} data loaded from cache: {phase.shape}")
                
                if split == 'train':
                    train_phase = phase
                    train_labels = labels
                else:
                    test_phase = phase
                    test_labels = labels
            except Exception as e:
                print(f"⚠ Failed to load cache: {e}, will rebuild...")
                use_cache = False
        
        if not use_cache or not os.path.exists(cache_path):
            # 构建缓存
            if split == 'train' or split == 'test':  # 确保目录存在
                if os.path.exists(split_dir):
                    print(f"[INFO] Building cache for {split}...")
                    builder = PhaseCacheBuilder(data_dir, verbose=True)
                    cache_path, success = builder.build_cache(split, force_rebuild=True)
                    
                    if success and os.path.exists(cache_path):
                        cache = np.load(cache_path)
                        phase = cache['phase'].astype(np.float32)
                        labels = cache['labels'].astype(np.int64)
                        
                        if split == 'train':
                            train_phase = phase
                            train_labels = labels
                        else:
                            test_phase = phase
                            test_labels = labels
    
    if train_phase is None:
        raise FileNotFoundError(f"❌ Failed to load or build phase data from {data_dir}")
    
    return train_phase, train_labels, test_phase, test_labels


def normalize_phase_data(phase_data):
    """
    Normalize phase data to [-1, 1] range.
    
    输入: phase_data, shape: (..., 256, 256), range: [-π, π]
    输出: normalized, shape: (..., 1, 256, 256), range: [-1, 1]
    """
    # 相位范围: [-π, π] → [-1, 1]
    phase_norm = phase_data / np.pi
    # 添加通道维度 (1, 256, 256)
    phase_norm = np.expand_dims(phase_norm, axis=1)
    return phase_norm.astype(np.float32)


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
        name='conditional_dit_phase_hologram',  # 更新为相位全息图
        config={
            'num_epochs': 100,
            'batch_size': 128,
            'learning_rate': 1e-4,
            'timesteps': 1000,
            'model_type': 'ConditionalDiT',
            'data_type': 'phase_hologram',  # 新增
            'img_size': 256,                # 更新
            'patch_size': 64,               # 更新
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
    
    # Load phase hologram data
    print("Loading phase hologram data...")
    train_phase, train_labels, test_phase, test_labels = load_phase_data('data/minst_phase')
    print(f"✅ Phase data loaded successfully!")
    
    # Normalize phase to [-1, 1]
    train_phase = normalize_phase_data(train_phase)
    if test_phase is not None:
        test_phase = normalize_phase_data(test_phase)
    
    print(f"Train phase shape: {train_phase.shape}, dtype: {train_phase.dtype}")
    print(f"  Range: [{train_phase.min():.4f}, {train_phase.max():.4f}]")
    print(f"Train labels shape: {train_labels.shape}, dtype: {train_labels.dtype}")
    
    # Create dataset
    train_dataset = TensorDataset(
        torch.from_numpy(train_phase).float(),
        torch.from_numpy(train_labels).long()
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Create model and diffusion
    print("Creating model...")
    # 注意: 相位数据是256×256, 而不是28×28
    # 使用64×64 patch (4x4 patches = 16个token)
    model = ConditionalDiT(
        img_size=256,           # 改为256 (相位数据分辨率)
        patch_size=64,          # 改为64 (256/64 = 4, 共16个token)
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
