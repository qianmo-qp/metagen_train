"""
Inference script: Generate phase hologram samples using epoch 140 model (best quality).

This script loads the best-performing checkpoint (epoch 140, loss=0.337) 
and generates conditional phase hologram samples.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import argparse

from dit_model import ConditionalDiT
from diffusion import create_diffusion


def load_checkpoint(checkpoint_path, model, device):
    """Load model from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle DDP-wrapped model state dicts
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if isinstance(state_dict, dict):
        # Remove 'module.' prefix if present (from DDP)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict)
    print(f"✅ Loaded checkpoint: {checkpoint_path}")
    return model


def generate_phase_samples(
    model,
    diffusion,
    device,
    num_classes=10,
    batch_size=1,
    timesteps=1000,
    method='ddim',
    num_steps=50,
):
    """
    Generate phase hologram samples for each digit class (0-9).
    
    Args:
        model: Trained ConditionalDiT model
        diffusion: Diffusion schedule
        device: 'cuda' or 'cpu'
        num_classes: 10 digit classes
        batch_size: Samples per class
        timesteps: Total diffusion timesteps (1000)
        method: 'ddim' (fast) or 'ddpm' (high-quality)
        num_steps: DDIM steps (default 50)
    
    Returns:
        samples: (num_classes * batch_size, 1, 256, 256) phase values in [-π, π]
        labels: (num_classes * batch_size,) digit labels
    """
    model.eval()
    all_samples = []
    all_labels = []
    
    with torch.no_grad():
        for class_id in range(num_classes):
            print(f"  Generating {batch_size} sample(s) for digit {class_id}...")
            
            class_labels = torch.full((batch_size,), class_id, dtype=torch.long, device=device)
            
            if method == 'ddim':
                # Fast DDIM sampling
                samples = diffusion.sample_ddim(
                    model,
                    num_samples=batch_size,
                    num_classes=10,
                    device=device,
                    num_steps=num_steps,
                    eta=0.0,  # Deterministic
                    class_labels=class_labels
                )
            else:
                # Full DDPM sampling (slow but highest quality)
                samples = torch.randn(batch_size, 1, 256, 256, device=device)
                
                for t in reversed(range(timesteps)):
                    if t % 100 == 0:
                        print(f"    Step {t}/{timesteps}")
                    
                    t_tensor = torch.full((batch_size,), t, dtype=torch.long, device=device)
                    
                    samples = diffusion.p_sample(
                        model, samples, t_tensor, class_labels, clip_denoised=True
                    )
            
            all_samples.append(samples.cpu())
            all_labels.extend([class_id] * batch_size)
    
    samples = torch.cat(all_samples, dim=0)
    labels = torch.tensor(all_labels, dtype=torch.long)
    
    return samples, labels


def normalize_phase(phase_values):
    """
    Normalize phase values from [-π, π] to [0, 1] for visualization.
    
    Args:
        phase_values: Tensor in [-π, π]
    
    Returns:
        Normalized tensor in [0, 1]
    """
    return (phase_values + np.pi) / (2 * np.pi)


def visualize_phase_holograms(samples, labels, output_path='generated_phases.png', nrow=10):
    """
    Visualize phase hologram samples as grayscale images.
    
    Args:
        samples: (N, 1, 256, 256) phase values in [-π, π]
        labels: (N,) digit labels
        output_path: Where to save the visualization
        nrow: Images per row
    """
    # Normalize to [0, 1]
    samples_norm = normalize_phase(samples)
    samples_norm = torch.clamp(samples_norm, 0.0, 1.0)
    
    num_samples = samples.shape[0]
    num_cols = min(nrow, num_samples)
    num_rows = (num_samples + nrow - 1) // nrow
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
    
    if num_rows == 1:
        axes = axes.reshape(1, -1)
    if num_cols == 1:
        axes = axes.reshape(-1, 1)
    
    axes = axes.flatten()
    
    for idx in range(num_samples):
        ax = axes[idx]
        phase = samples_norm[idx, 0].numpy()
        ax.imshow(phase, cmap='twilight_shifted')
        ax.set_title(f'Digit {labels[idx].item()}')
        ax.axis('off')
    
    # Hide remaining axes
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved visualization: {output_path}")
    plt.close()


def save_phase_npy(samples, labels, output_dir='outputs/inference_epoch140'):
    """Save generated phases as NPY files for further processing."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    for class_id in range(10):
        class_mask = labels == class_id
        class_samples = samples[class_mask]  # (batch_size, 1, 256, 256)
        
        for idx, sample in enumerate(class_samples):
            phase = sample[0].numpy()  # (256, 256)
            filename = os.path.join(output_dir, f'phase_digit{class_id}_{idx:03d}.npy')
            np.save(filename, phase)
    
    print(f"✅ Saved phase data to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description='Generate phase hologram samples using epoch 140 checkpoint (best quality)'
    )
    parser.add_argument(
        '--checkpoint',
        default='/mnt/model_data/qp/metagen_train/checkpoints/checkpoint_epoch_140.pt',
        help='Path to checkpoint file'
    )
    parser.add_argument(
        '--method',
        choices=['ddim', 'ddpm'],
        default='ddim',
        help='Sampling method: ddim (fast, ~30s) or ddpm (slow, ~10m, highest quality)'
    )
    parser.add_argument(
        '--num-steps',
        type=int,
        default=50,
        help='Number of DDIM steps (default: 50, range: 10-200)'
    )
    parser.add_argument(
        '--batch-per-class',
        type=int,
        default=2,
        help='Number of samples to generate per digit class (0-9)'
    )
    parser.add_argument(
        '--output-dir',
        default='outputs/inference_epoch140',
        help='Output directory for generated samples'
    )
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 Device: {device}")
    print(f"📋 Sampling method: {args.method.upper()} ({args.num_steps} steps)")
    print(f"📦 Batch size: {args.batch_per_class} sample(s) per digit class")
    
    # Model configuration (114M params, 256×256 phase)
    model_config = {
        'img_size': 256,
        'patch_size': 8,
        'in_channels': 1,
        'hidden_dim': 768,
        'num_heads': 12,
        'num_layers': 12,
        'time_dim': 256,
        'num_classes': 10,
        'mlp_ratio': 4,
    }
    
    print("\n📐 Creating model (114M params)...")
    model = ConditionalDiT(**model_config)
    model.to(device)
    
    # Load checkpoint
    print(f"\n🔌 Loading checkpoint: {args.checkpoint}")
    model = load_checkpoint(args.checkpoint, model, device)
    
    # Create diffusion
    print("⚙️  Creating diffusion schedule...")
    diffusion = create_diffusion(timesteps=1000, schedule_type='cosine')
    for attr_name in dir(diffusion):
        attr = getattr(diffusion, attr_name)
        if isinstance(attr, torch.Tensor):
            setattr(diffusion, attr_name, attr.to(device))
    
    # Generate samples
    print(f"\n🎨 Generating {args.batch_per_class} sample(s) per digit class...")
    samples, labels = generate_phase_samples(
        model,
        diffusion,
        device,
        num_classes=10,
        batch_size=args.batch_per_class,
        timesteps=1000,
        method=args.method,
        num_steps=args.num_steps,
    )
    
    # Save outputs
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Visualization
    vis_path = os.path.join(args.output_dir, 'phase_visualization.png')
    visualize_phase_holograms(samples, labels, vis_path, nrow=10)
    
    # NPY files for optical reconstruction
    save_phase_npy(samples, labels, args.output_dir)
    
    # Statistics
    print(f"\n📊 Generation statistics:")
    print(f"  Total samples: {samples.shape[0]}")
    print(f"  Shape: {samples.shape}")
    print(f"  Phase range: [{samples.min():.3f}, {samples.max():.3f}]")
    print(f"  Mean: {samples.mean():.3f}, Std: {samples.std():.3f}")
    
    print(f"\n✅ Complete! Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
