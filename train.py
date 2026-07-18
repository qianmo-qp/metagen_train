"""
Training loop for Conditional DiT on MNIST with W&B monitoring.
"""

import os
import sys
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
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


def setup_distributed():
    """Initialize DDP process group and return rank, world_size, local_rank."""
    if 'RANK' not in os.environ:
        # Single GPU mode
        return 0, 1, 0
    
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up DDP process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process (rank 0)."""
    return not dist.is_initialized() or dist.get_rank() == 0


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
        learning_rate,
        num_epochs=10,
        batch_size=128,
        checkpoint_dir='checkpoints',
        log_interval=100,
        gradient_accumulation_steps=4,
        use_wandb=False,
        sample_interval=10,
        warmup_epochs=3,
        flat_epochs=97,
        num_batches_per_epoch=938,
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
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.learning_rate = learning_rate
        
        # Check if model is wrapped in DDP
        self.is_ddp = isinstance(model, DDP)
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        self.loss_fn = nn.MSELoss()
        
        # Create checkpoint directory
        Path(self.checkpoint_dir).mkdir(exist_ok=True)
        
        self.step = 0
        self.losses = []
        self.logger = logging.getLogger(__name__)
        self.best_loss = float('inf')
        self.best_checkpoint_path = None
        self.nan_count = 0  # Track consecutive NaN loss occurrences
        self.nan_grad_count = 0  # Track consecutive NaN gradient norm occurrences
        
        # LR Scheduler: Linear warmup + Flat + Cosine decay
        # Calculate actual optimizer steps
        steps_per_epoch = num_batches_per_epoch // gradient_accumulation_steps
        self.total_steps = num_epochs * steps_per_epoch
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.flat_steps = flat_epochs * steps_per_epoch
        self.decay_start = self.warmup_steps + self.flat_steps  # Cosine decay starts here
        
        # Use LambdaLR for warmup + flat + cosine decay
        def lr_lambda(current_step):
            if current_step < self.warmup_steps:
                # Linear warmup from 0.1x to 1.0x
                return 0.1 + 0.9 * (current_step / self.warmup_steps)
            elif current_step < self.decay_start:
                # Flat phase: keep LR at peak
                return 1.0
            else:
                # Cosine decay from 1.0x to eta_min_ratio
                progress = (current_step - self.decay_start) / max(1, self.total_steps - self.decay_start)
                eta_min_ratio = 1e-6 / learning_rate  # Decay to 1e-6
                return eta_min_ratio + (1.0 - eta_min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        if is_main_process():
            print(f"LR Schedule: warmup {warmup_epochs} epochs ({self.warmup_steps} steps) "
                  f"+ flat {flat_epochs} epochs ({self.flat_steps} steps) "
                  f"+ cosine decay {num_epochs - warmup_epochs - flat_epochs} epochs")
            print(f"  Peak LR: {learning_rate}, Min LR: 1e-6")
            print(f"  Total optimizer steps: {self.total_steps}")
    
    def get_model(self):
        """Get the underlying model (unwrap DDP if needed)."""
        return self.model.module if self.is_ddp else self.model
    
    def train_epoch(self, train_loader):
        """Train for one epoch with NaN detection and gradient monitoring."""
        self.model.train()
        epoch_loss = 0.0
        valid_batches = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Sample random timesteps
            t = torch.randint(0, self.diffusion.timesteps, (images.shape[0],), device=self.device)
            
            # Forward diffusion: add noise
            x_t, noise = self.diffusion.q_sample(images, t)
            
            # Predict noise (scaled loss for gradient accumulation)
            if (batch_idx + 1) % self.gradient_accumulation_steps == 1 or self.gradient_accumulation_steps == 1:
                self.optimizer.zero_grad()
            noise_pred = self.model(x_t, t, labels)
            
            # Compute loss (scaled for gradient accumulation)
            loss = self.loss_fn(noise_pred, noise) / self.gradient_accumulation_steps
            
            # NaN detection - skip this batch if loss is NaN
            if torch.isnan(loss) or torch.isinf(loss):
                self.nan_count += 1
                if is_main_process():
                    print(f"⚠️  NaN/Inf loss detected at step {self.step}, batch {batch_idx + 1} (count: {self.nan_count})")
                self.optimizer.zero_grad()  # Clear any accumulated gradients
                if self.nan_count >= 5:
                    if is_main_process():
                        print(f"🛑 {self.nan_count} consecutive NaN losses! Halving LR and loading best checkpoint...")
                    self._recover_from_nan()
                self.step += 1
                continue
            
            self.nan_count = 0  # Reset loss NaN counter on valid loss
            loss.backward()
            unscaled_loss = loss.item() * self.gradient_accumulation_steps
            
            # Gradient clipping & optimizer step every N steps
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                # Check gradient norm before stepping
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Skip update if gradient is NaN/Inf
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    self.nan_grad_count += 1
                    if self.nan_grad_count % 50 == 1 and is_main_process():
                        print(f"⚠️  NaN/Inf gradient norm at step {self.step} (consecutive: {self.nan_grad_count}), skipping update")
                    self.optimizer.zero_grad()
                    
                    # Trigger recovery on consecutive NaN gradients
                    if self.nan_grad_count >= 20:
                        if is_main_process():
                            print(f"🛑 {self.nan_grad_count} consecutive NaN gradients! Loading best checkpoint and halving LR...")
                        self._recover_from_nan()
                else:
                    self.nan_grad_count = 0  # Reset gradient NaN counter on valid grad
                    self.optimizer.step()
                    self.scheduler.step()  # Update LR after each valid step
            
            epoch_loss += unscaled_loss
            valid_batches += 1
            self.losses.append(unscaled_loss)
            self.step += 1
            
            if (batch_idx + 1) % self.log_interval == 0:
                avg_loss = epoch_loss / max(valid_batches, 1)
                current_lr = self.optimizer.param_groups[0]['lr']
                # Only print from rank 0
                if is_main_process():
                    print(f"Step {self.step}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {avg_loss:.6f}, LR: {current_lr:.2e}")
                
                # Log to W&B (only from rank 0)
                if self.use_wandb and is_main_process():
                    wandb.log({
                        "loss": avg_loss,
                        "learning_rate": current_lr,
                        "grad_norm": grad_norm.item() if not (torch.isnan(grad_norm) or torch.isinf(grad_norm)) else 0,
                        "step": self.step,
                        "batch": batch_idx + 1,
                        "epoch": int(self.step / len(train_loader)) + 1,
                    }, step=self.step)
        
        return epoch_loss / max(valid_batches, 1)
    
    def _recover_from_nan(self):
        """Recover from NaN by loading best checkpoint and reducing LR."""
        best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
        if os.path.exists(best_path):
            checkpoint = torch.load(best_path, map_location=self.device)
            self.get_model().load_state_dict(checkpoint['model_state_dict'])
        # Reduce LR by 30% (not halve — avoid LR collapsing too fast)
        self.scheduler.base_lrs = [lr * 0.7 for lr in self.scheduler.base_lrs]
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.7
        if is_main_process():
            if os.path.exists(best_path):
                print(f"✅ Restored from best checkpoint (epoch {checkpoint.get('epoch', '?')}, loss={checkpoint.get('loss', '?')})")
            print(f"   New base LR: {self.scheduler.base_lrs[0]:.2e}, current LR: {self.optimizer.param_groups[0]['lr']:.2e}")
        self.nan_count = 0
        self.nan_grad_count = 0
    
    def save_checkpoint(self, epoch, loss=None, is_best=False):
        """Save model checkpoint — keep only latest and best."""
        checkpoint = {
            'epoch': epoch,
            'step': self.step,
            'model_state_dict': self.get_model().state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self, 'scheduler') else None,
            'losses': self.losses,
            'loss': loss,
        }
        
        # Always overwrite latest checkpoint
        latest_path = os.path.join(self.checkpoint_dir, 'checkpoint_latest.pt')
        torch.save(checkpoint, latest_path)
        if is_main_process():
            print(f"Saved checkpoint (epoch {epoch}) to {latest_path}")
        
        # Save best checkpoint when loss improves
        is_new_best = loss is not None and loss < self.best_loss
        if is_new_best or is_best:
            self.best_loss = loss if loss is not None else self.best_loss
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            if is_main_process():
                tag = "🏆 New best" if is_new_best else "Best"
                print(f"{tag} model saved (loss={self.best_loss:.6f})")
    
    def generate_samples(self, epoch):
        """Generate recovered digit samples via DDPM and log to W&B."""
        import io
        from sample_utils import generate_samples_ddpm, make_digit_grid
        
        self.logger.info(f"Generating samples at epoch {epoch} (DDPM 1000 steps)...")
        self.model.eval()
        
        # Generate 1 sample per class (0-9)
        class_labels = torch.arange(10, device=self.device)
        phase_samples = generate_samples_ddpm(
            self.get_model(), self.diffusion, self.device, class_labels
        )
        
        # Render 2x5 grid
        title = f'Epoch {epoch} — Recovered Digits (DDPM 1000 steps)'
        fig = make_digit_grid(phase_samples, list(range(10)), title=title, ncols=5)
        
        # Save locally
        sample_path = os.path.join('outputs', f'recovered_digits_epoch_{epoch}.png')
        Path('outputs').mkdir(exist_ok=True)
        fig.savefig(sample_path, dpi=100, bbox_inches='tight')
        self.logger.info(f"Saved recovered digits to {sample_path}")
        
        # Log to W&B
        if self.use_wandb:
            import matplotlib.pyplot as plt
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            from PIL import Image as PILImage
            pil_image = PILImage.open(buf)
            pil_image.load()
            wandb.log({
                f"recovered_digits_epoch_{epoch}": wandb.Image(pil_image.copy()),
                "epoch": epoch,
            }, step=self.step)
            self.logger.info(f"Logged recovered digits to W&B for epoch {epoch}")
        
        import matplotlib.pyplot as plt
        plt.close(fig)
        self.model.train()
    
    def train(self, train_loader, sampler=None):
        """Train for multiple epochs."""
        if is_main_process():
            print(f"Training on {self.device}")
            print(f"Total epochs: {self.num_epochs}")
        
        for epoch in range(self.num_epochs):
            # Set epoch for DDP sampler (important for shuffling)
            if sampler is not None and hasattr(sampler, 'set_epoch'):
                sampler.set_epoch(epoch)
            
            if is_main_process():
                print(f"\n=== Epoch {epoch + 1}/{self.num_epochs} ===")
            avg_loss = self.train_epoch(train_loader)
            if is_main_process():
                print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.6f}")
            
            # Save checkpoint (only from rank 0)
            if is_main_process():
                self.save_checkpoint(epoch + 1, loss=avg_loss)
            
            # Generate samples every sample_interval epochs (only from rank 0)
            if (epoch + 1) % self.sample_interval == 0 and is_main_process():
                self.logger.info(f"Sampling at epoch {epoch + 1}")
                try:
                    self.generate_samples(epoch + 1)
                except Exception as e:
                    import traceback
                    self.logger.error(f"Error during sampling: {e}")
                    self.logger.error(f"Traceback:\n{traceback.format_exc()}")
            
            # Log epoch metrics to W&B (only from rank 0)
            if self.use_wandb and is_main_process():
                wandb.log({
                    "epoch_loss": avg_loss,
                    "epoch": epoch + 1,
                })
        
        if is_main_process():
            print("\nTraining complete!")
        return self.losses


def main():
    # Reduce CUDA memory fragmentation for large models
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Initialize distributed training
    rank, world_size, local_rank = setup_distributed()
    is_distributed = world_size > 1
    
    if is_distributed:
        # Set device for this GPU
        torch.cuda.set_device(local_rank)
        device = f'cuda:{local_rank}'
        
        # Only rank 0 prints info
        if rank == 0:
            print(f"✅ DDP initialized: {world_size} GPUs")
            for i in range(world_size):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"   Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory * world_size / 1e9:.2f} GB")
    else:
        # Single GPU mode
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
    
    if rank == 0:
        print(f"\nUsing device: {device}")
    
    # Hyperparameters
    num_epochs = 550
    batch_size_per_gpu = 48 if is_distributed else 64  # A40 48G: 48×3=144, Single: 64
    gradient_accumulation_steps = 1
    learning_rate = 3e-5  # 0703验证稳定140epoch的峰值LR
    warmup_epochs = 5     # Linear warmup for first 5 epochs (更平缓)
    flat_epochs = 350     # 保持峰值LR不变的epoch数（warmup+flat共255 epochs）
    timesteps = 1000
    sample_interval = 20  # Generate recovered digits every 20 epochs
    
    # Initialize W&B (only from rank 0)
    if rank == 0:
        print("\n📊 Initializing W&B...")
        wandb.init(
            project='minst',
            name='conditional_dit_phase_hologram_ddp' if is_distributed else 'conditional_dit_phase_hologram',
            config={
                'num_epochs': num_epochs,
                'batch_size_per_gpu': batch_size_per_gpu,
                'world_size': world_size,
                'effective_batch_size': batch_size_per_gpu * world_size,
                'gradient_accumulation_steps': gradient_accumulation_steps,
                'learning_rate': learning_rate,
                'warmup_epochs': warmup_epochs,
                'flat_epochs': flat_epochs,
                'decay_epochs': num_epochs - warmup_epochs - flat_epochs,
                'timesteps': timesteps,
                'model_type': 'ConditionalDiT',
                'data_type': 'phase_hologram',
                'img_size': 256,
                'patch_size': 8,
                'hidden_dim': 768,
                'num_layers': 12,
                'num_heads': 12,
            }
        )
        print("✅ W&B initialized")
    
    # Load phase hologram data (only from rank 0 to avoid disk contention)
    if rank == 0:
        print("Loading phase hologram data...")
    train_phase, train_labels, test_phase, test_labels = load_phase_data('data/minst_phase')
    if rank == 0:
        print(f"✅ Phase data loaded successfully!")
    
    # Normalize phase to [-1, 1]
    train_phase = normalize_phase_data(train_phase)
    if test_phase is not None:
        test_phase = normalize_phase_data(test_phase)
    
    if rank == 0:
        print(f"Train phase shape: {train_phase.shape}, dtype: {train_phase.dtype}")
        print(f"  Range: [{train_phase.min():.4f}, {train_phase.max():.4f}]")
        print(f"Train labels shape: {train_labels.shape}, dtype: {train_labels.dtype}")
    
    # Create dataset and dataloader
    train_dataset = TensorDataset(
        torch.from_numpy(train_phase).float(),
        torch.from_numpy(train_labels).long()
    )
    
    # Use DistributedSampler for DDP
    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size_per_gpu,
            sampler=train_sampler,
            num_workers=0,
            pin_memory=True,
        )
    else:
        train_sampler = None
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size_per_gpu,
            shuffle=True,
            num_workers=0
        )
    
    if rank == 0:
        print(f"Batch size per GPU: {batch_size_per_gpu}")
        print(f"Total batch size: {batch_size_per_gpu * world_size * gradient_accumulation_steps}")
        print(f"World size: {world_size}, Gradient accumulation: {gradient_accumulation_steps}")
    
    # Create model and diffusion
    if rank == 0:
        print("Creating model...")
    model = ConditionalDiT(
        img_size=256,
        patch_size=8,
        in_channels=1,
        hidden_dim=768,
        num_heads=12,
        num_layers=12,
        time_dim=256,
        num_classes=10,
        mlp_ratio=4,
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"Model parameters: {num_params:,}")
    
    diffusion = create_diffusion(timesteps=timesteps, schedule_type='cosine')
    
    # Move diffusion buffers to device
    for attr_name in dir(diffusion):
        attr = getattr(diffusion, attr_name)
        if isinstance(attr, torch.Tensor):
            setattr(diffusion, attr_name, attr.to(device))
    
    # Wrap model in DDP if distributed
    if is_distributed:
        model = DDP(model.to(device), device_ids=[local_rank], find_unused_parameters=True)
        if rank == 0:
            print(f"✅ Model wrapped in DDP")
    
    # Train
    trainer = Trainer(
        model=model,
        diffusion=diffusion,
        device=device,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        batch_size=batch_size_per_gpu,
        gradient_accumulation_steps=gradient_accumulation_steps,
        log_interval=100,
        use_wandb=(rank == 0),  # Only rank 0 logs to W&B
        sample_interval=sample_interval,
        warmup_epochs=warmup_epochs,
        flat_epochs=flat_epochs,
        num_batches_per_epoch=len(train_loader),
    )
    
    losses = trainer.train(train_loader, sampler=train_sampler)
    
    # Cleanup
    if rank == 0:
        print(f"\nTraining complete! Final checkpoint saved.")
        print(f"Total steps: {trainer.step}")
        wandb.finish()
        print("\n📊 W&B run finished")
    
    cleanup_distributed()


if __name__ == "__main__":
    main()
