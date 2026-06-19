"""
Sampling script: Generate conditional MNIST digits using trained DiT model.
"""

import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

from dit_model import ConditionalDiT
from diffusion import create_diffusion


def load_checkpoint(checkpoint_path, model, device):
    """Load model from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from {checkpoint_path}")
    return model


def generate_samples(
    model,
    diffusion,
    device,
    num_classes=10,
    batch_size=10,
    timesteps=1000,
):
    """
    Generate samples for each class.
    
    Args:
        model: Trained DiT model
        diffusion: Diffusion schedule
        device: Device to use
        num_classes: Number of classes (0-9)
        batch_size: Samples per class
    Returns:
        samples: (num_classes * batch_size, 1, 28, 28) tensor
        labels: (num_classes * batch_size,) tensor
    """
    model.eval()
    
    all_samples = []
    all_labels = []
    
    for class_id in range(num_classes):
        print(f"Generating {batch_size} samples for class {class_id}...")
        
        # Class labels for this batch
        class_labels = torch.full((batch_size,), class_id, dtype=torch.long, device=device)
        
        # Start from pure noise
        x_t = torch.randn(batch_size, 1, 28, 28, device=device)
        
        # Reverse diffusion
        for t in reversed(range(timesteps)):
            if t % 100 == 0:
                print(f"  Reverse step {t}/{timesteps}")
            
            t_tensor = torch.full((batch_size,), t, dtype=torch.long, device=device)
            
            with torch.no_grad():
                x_t = diffusion.p_sample(model, x_t, t_tensor, class_labels, clip_denoised=True)
        
        all_samples.append(x_t.cpu())
        all_labels.extend([class_id] * batch_size)
    
    samples = torch.cat(all_samples, dim=0)
    labels = torch.tensor(all_labels, dtype=torch.long)
    
    return samples, labels


def denormalize(x):
    """Convert from [-1, 1] to [0, 1]."""
    return (x + 1.0) / 2.0


def save_grid_image(samples, labels, output_path='generated_samples.png', nrow=10):
    """
    Save samples as a grid image.
    
    Args:
        samples: (N, 1, 28, 28) tensor in [-1, 1]
        labels: (N,) tensor with class labels
        output_path: Path to save the grid image
        nrow: Number of images per row
    """
    # Denormalize to [0, 1]
    samples = denormalize(samples)
    samples = torch.clamp(samples, 0.0, 1.0)
    
    # Create grid
    num_samples = samples.shape[0]
    num_cols = nrow
    num_rows = (num_samples + nrow - 1) // nrow
    
    # Create figure
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
    
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    
    axes = axes.flatten()
    
    for idx in range(num_samples):
        ax = axes[idx]
        img = samples[idx, 0].numpy()
        ax.imshow(img, cmap='gray')
        ax.set_title(f'Class {labels[idx].item()}')
        ax.axis('off')
    
    # Hide remaining axes
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"Saved grid image to {output_path}")
    plt.close()


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Model configuration
    model_config = {
        'img_size': 28,
        'patch_size': 4,
        'in_channels': 1,
        'hidden_dim': 192,
        'num_heads': 3,
        'num_layers': 6,
        'time_dim': 256,
        'num_classes': 10,
        'mlp_ratio': 4,
    }
    
    # Create model
    print("Creating model...")
    model = ConditionalDiT(**model_config)
    
    # Load checkpoint
    checkpoint_path = 'checkpoints/best_model.pt'
    if not os.path.exists(checkpoint_path):
        # Try latest checkpoint
        checkpoint_dir = 'checkpoints'
        checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint_epoch_')])
        if checkpoints:
            checkpoint_path = os.path.join(checkpoint_dir, checkpoints[-1])
            print(f"No best_model.pt found, using latest: {checkpoint_path}")
        else:
            raise FileNotFoundError("No checkpoints found in 'checkpoints' directory")
    
    model = load_checkpoint(checkpoint_path, model, device)
    model.to(device)
    
    # Create diffusion
    diffusion = create_diffusion(timesteps=1000, schedule_type='cosine')
    
    # Move diffusion buffers to device
    for attr_name in dir(diffusion):
        attr = getattr(diffusion, attr_name)
        if isinstance(attr, torch.Tensor):
            setattr(diffusion, attr_name, attr.to(device))
    
    # Generate samples: 1 sample per class
    print("\n=== Generating 1 sample per class (0-9) ===")
    samples, labels = generate_samples(
        model,
        diffusion,
        device,
        num_classes=10,
        batch_size=1,
        timesteps=1000,
    )
    
    # Save grid
    output_dir = 'outputs'
    Path(output_dir).mkdir(exist_ok=True)
    output_path = os.path.join(output_dir, 'generated_samples_1per_class.png')
    save_grid_image(samples, labels, output_path, nrow=10)
    
    # Generate more samples for better visualization
    print("\n=== Generating 10 samples per class ===")
    samples_many, labels_many = generate_samples(
        model,
        diffusion,
        device,
        num_classes=10,
        batch_size=10,
        timesteps=1000,
    )
    
    output_path_many = os.path.join(output_dir, 'generated_samples_10per_class.png')
    save_grid_image(samples_many, labels_many, output_path_many, nrow=10)
    
    print("\n=== Sampling complete! ===")
    print(f"Outputs saved to '{output_dir}' directory")


if __name__ == "__main__":
    main()
