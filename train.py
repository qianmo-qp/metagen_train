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


def load_phase_data(data_dir='data/minst_phase'):
    """
    Load phase hologram data from chunked NPZ files.
    
    文件结构:
    data_dir/
    ├── train/
    │   ├── minst_phase_train_01.npz (样本 0-9,999)
    │   ├── minst_phase_train_02.npz (样本 10,000-19,999)
    │   ├── minst_phase_train_03.npz (样本 20,000-29,999)
    │   ├── minst_phase_train_04.npz (样本 30,000-39,999)
    │   ├── minst_phase_train_05.npz (样本 40,000-49,999)
    │   └── minst_phase_train_06.npz (样本 50,000-59,999)
    └── test/
        └── minst_phase_test_01.npz (样本 0-9,999)
    """
    # Handle both relative and absolute paths
    if not os.path.isabs(data_dir) and not os.path.exists(data_dir):
        # Try server path if relative path doesn't exist
        server_path = f'/mnt/model_data/qp/metagen_train/{data_dir}'
        if os.path.exists(server_path):
            data_dir = server_path
            print(f"[INFO] Using server path: {data_dir}")
    
    # 加载分块 NPZ 文件
    train_phase, train_labels = _load_chunked_npz(data_dir, 'train')
    test_phase, test_labels = _load_chunked_npz(data_dir, 'test')
    
    if train_phase is None:
        raise FileNotFoundError(f"❌ Failed to load train phase data from {data_dir}")
    
    return train_phase, train_labels, test_phase, test_labels


def _load_chunked_npz(data_dir, split):
    """
    加载分块的 NPZ 文件（如果使用了优化版本的处理器）
    
    参数:
        data_dir: 数据目录
        split: 'train' 或 'test'
    
    返回:
        phase_data: (N, 256, 256) float32 数组
        labels_data: (N,) int64 数组
    """
    split_dir = os.path.join(data_dir, split)
    
    if not os.path.exists(split_dir):
        print(f"❌ Directory not found: {split_dir}")
        return None, None
    
    # 查找所有分块 NPZ 文件
    npz_files = sorted([f for f in os.listdir(split_dir) 
                       if f.startswith(f'minst_phase_{split}_') and f.endswith('.npz')])
    
    if not npz_files:
        print(f"❌ No NPZ files found in: {split_dir}")
        return None, None
    
    print(f"\n[INFO] Found {len(npz_files)} chunked NPZ files for {split}")
    
    # 加载并合并所有分块
    all_phases = []
    all_labels = []
    total_size_mb = 0
    total_samples = 0
    
    for idx, npz_file in enumerate(npz_files, 1):
        npz_path = os.path.join(split_dir, npz_file)
        file_size_mb = os.path.getsize(npz_path) / (1024 ** 2)
        total_size_mb += file_size_mb
        
        try:
            print(f"  [{idx}/{len(npz_files)}] Loading {npz_file} ({file_size_mb:.1f} MB)...", end='', flush=True)
            
            data = np.load(npz_path, allow_pickle=False)
            phase = data['phase'].astype(np.float32)
            labels = data['labels'].astype(np.int64)
            
            samples = phase.shape[0]
            total_samples += samples
            
            all_phases.append(phase)
            all_labels.append(labels)
            
            # 显示详细信息
            phase_min, phase_max, phase_mean = phase.min(), phase.max(), phase.mean()
            print(f" ✓ {phase.shape} (labels: {labels.min()}-{labels.max()}, "
                  f"phase: [{phase_min:.3f}, {phase_max:.3f}], mean: {phase_mean:.3f})")
        except Exception as e:
            print(f" ❌ Error: {e}")
            return None, None
    
    # 合并所有分块
    if all_phases:
        print(f"\n[INFO] Merging {len(all_phases)} chunks ({total_size_mb:.1f} MB total)...", end='', flush=True)
        phase_data = np.concatenate(all_phases, axis=0)
        labels_data = np.concatenate(all_labels, axis=0)
        print(f" ✓ Done")
        
        # 显示最终统计
        print(f"✓ {split.upper()} data loaded:")
        print(f"  Shape: {phase_data.shape}")
        print(f"  Dtype: {phase_data.dtype}")
        print(f"  Labels dtype: {labels_data.dtype}")
        print(f"  Labels range: {labels_data.min()}-{labels_data.max()}")
        print(f"  Phase range: [{phase_data.min():.3f}, {phase_data.max():.3f}]")
        print(f"  Phase mean: {phase_data.mean():.3f}")
        
        return phase_data, labels_data
    
    return None, None


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
            'data_type': 'phase_hologram',
            'img_size': 256,
            'patch_size': 8,
            'hidden_dim': 384,
            'num_layers': 12,
            'num_heads': 6,
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
    # 相位数据 256×256, patch_size=8 → 32×32=1024 tokens
    # 增大模型容量: hidden_dim=384, num_layers=12, num_heads=6
    model = ConditionalDiT(
        img_size=256,
        patch_size=8,            # 256/8 = 32, 32×32 = 1024 tokens
        in_channels=1,
        hidden_dim=384,          # 192 → 384
        num_heads=6,             # 3 → 6
        num_layers=12,           # 6 → 12
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
