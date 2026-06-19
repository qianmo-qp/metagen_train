"""
Training loop for Conditional DiT on MNIST.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path

# Disable torch._dynamo to avoid ONNX import issues
os.environ['TORCH_DISABLE_DYNANMO'] = '1'

from dit_model import ConditionalDiT
from diffusion import create_diffusion


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
    ):
        self.model = model.to(device)
        self.diffusion = diffusion
        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir
        self.log_interval = log_interval
        
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Create checkpoint directory
        Path(self.checkpoint_dir).mkdir(exist_ok=True)
        
        self.step = 0
        self.losses = []
    
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
    
    # Hyperparameters
    num_epochs = 10
    batch_size = 128
    learning_rate = 1e-4
    timesteps = 1000
    
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
    )
    
    losses = trainer.train(train_loader)
    
    print(f"\nTraining complete! Final checkpoint saved.")
    print(f"Total steps: {trainer.step}")


if __name__ == "__main__":
    main()
