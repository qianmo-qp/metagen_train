#!/usr/bin/env python
"""
Demo script: Quick training and sampling for Conditional DiT on MNIST.
This is a minimal example to verify everything works end-to-end.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path

from dit_model import ConditionalDiT
from diffusion import create_diffusion
from train import load_mnist_idx, normalize_images


def train_mini_demo(num_epochs=2, batch_size=256):
    """Mini training loop for testing."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🎯 Demo: Training Conditional DiT on MNIST")
    print(f"Device: {device}")
    
    # Load MNIST
    print("\n📊 Loading MNIST...")
    train_images = load_mnist_idx('data/minst/train-images.idx3-ubyte')
    train_labels = load_mnist_idx('data/minst/train-labels.idx1-ubyte')
    
    # Normalize
    train_images = normalize_images(train_images)
    
    # Use only first 5000 samples for fast demo
    train_images = train_images[:5000]
    train_labels = train_labels[:5000]
    
    dataset = TensorDataset(
        torch.from_numpy(train_images).float(),
        torch.from_numpy(train_labels).long()
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"✓ Loaded {len(train_images)} images")
    
    # Model
    print("\n🏗️  Creating model...")
    model = ConditionalDiT(
        img_size=28,
        patch_size=4,
        in_channels=1,
        hidden_dim=192,
        num_heads=3,
        num_layers=6,
        time_dim=256,
        num_classes=10,
    )
    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model has {num_params:,} parameters")
    
    # Diffusion & optimizer
    diffusion = create_diffusion(timesteps=1000, schedule_type='cosine')
    for attr_name in dir(diffusion):
        attr = getattr(diffusion, attr_name)
        if isinstance(attr, torch.Tensor):
            setattr(diffusion, attr_name, attr.to(device))
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()
    
    # Train
    print(f"\n🚀 Training for {num_epochs} epochs...")
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Noisy diffusion step
            t = torch.randint(0, diffusion.timesteps, (images.shape[0],), device=device)
            x_t, noise = diffusion.q_sample(images, t)
            
            # Predict noise
            optimizer.zero_grad()
            noise_pred = model(x_t, t, labels)
            loss = loss_fn(noise_pred, noise)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % 5 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"  Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(loader)}, Loss: {avg_loss:.6f}")
        
        print(f"✓ Epoch {epoch+1} - Avg Loss: {total_loss/len(loader):.6f}")
    
    # Save checkpoint
    Path('checkpoints').mkdir(exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, 'checkpoints/demo_checkpoint.pt')
    print(f"\n💾 Saved checkpoint to checkpoints/demo_checkpoint.pt")
    
    return model, diffusion


def sample_mini_demo(model, diffusion, device):
    """Generate samples from trained model."""
    
    print("\n🎨 Generating samples...")
    model.eval()
    
    # Generate 1 sample per class
    class_labels = torch.arange(10, device=device)
    x_t = torch.randn(10, 1, 28, 28, device=device)
    
    # Reverse diffusion with progress
    for t in reversed(range(0, 1000, 100)):  # Sample every 100 steps for speed
        t_tensor = torch.full((10,), t, dtype=torch.long, device=device)
        with torch.no_grad():
            x_t = diffusion.p_sample(model, x_t, t_tensor, class_labels, clip_denoised=True)
        print(f"  Step {t:4d} ✓")
    
    # Denormalize
    samples = (x_t + 1.0) / 2.0
    samples = torch.clamp(samples, 0.0, 1.0)
    
    # Save as image grid
    import matplotlib.pyplot as plt
    
    Path('outputs').mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()
    
    for idx in range(10):
        axes[idx].imshow(samples[idx, 0].cpu().numpy(), cmap='gray')
        axes[idx].set_title(f'Digit {idx}')
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('outputs/demo_generated.png', dpi=100, bbox_inches='tight')
    print(f"✓ Saved generated samples to outputs/demo_generated.png")
    plt.close()


def main():
    print("=" * 60)
    print("Conditional DiT - MNIST Demo")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Train
    model, diffusion = train_mini_demo(num_epochs=2, batch_size=256)
    
    # Sample
    sample_mini_demo(model, diffusion, device)
    
    print("\n" + "=" * 60)
    print("✅ Demo complete!")
    print("=" * 60)
    print("\n📝 Next steps:")
    print("  1. Review generated samples in outputs/demo_generated.png")
    print("  2. Run full training: python train.py")
    print("  3. Generate full samples: python sample.py")
    print("\n📖 Read QUICKSTART.md for more info")


if __name__ == "__main__":
    main()
