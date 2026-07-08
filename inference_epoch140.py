"""
Inference script: Generate phase hologram samples using epoch 140 model (best quality).

This script loads the best-performing checkpoint (epoch 140, loss=0.337) 
and generates conditional phase hologram samples with visualization.
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
                    # img_size will be auto-detected from model
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


def recover_image_from_phase(phase):
    """
    恢复图像: 从相位全息图恢复 MNIST 数字图像 (FFT反向变换)
    
    Args:
        phase: 相位数组 [-π, π], shape (256, 256)
    
    Returns:
        recovered: 恢复的图像 [0, 1], shape (256, 256)
    """
    # 相位 → 复数波前
    u1 = np.exp(1j * phase)
    
    # FFT 反向变换
    u2 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u1)))
    
    # 取幅度
    recovered = np.abs(u2)
    
    # 归一化到 [0, 1]
    recovered = recovered / (np.max(recovered) + 1e-8)
    
    return recovered


def phase_to_color_image(phase):
    """
    相位 → 彩色图 (twilight 色图)
    
    Args:
        phase: 相位数组 [-π, π], shape (256, 256)
    
    Returns:
        color_img: PIL Image (RGB)
    """
    fig, ax = plt.subplots(figsize=(5, 5), dpi=50)
    im = ax.imshow(phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 转为图像
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    return Image.fromarray(image)


def phase_to_recovered_image(phase):
    """
    相位 → 恢复数字图像 PNG
    
    Args:
        phase: 相位数组 [-π, π], shape (256, 256)
    
    Returns:
        img: PIL Image (L mode, grayscale)
    """
    recovered = recover_image_from_phase(phase)
    recovered_uint8 = (recovered * 255).astype(np.uint8)
    return Image.fromarray(recovered_uint8, mode='L')


def create_comparison_image(phase, digit_label):
    """
    创建对比图: 相位(彩色) + 相位(灰度16bit) + 恢复数字图像
    
    Args:
        phase: 相位数组 [-π, π]
        digit_label: 数字类别 (0-9)
    
    Returns:
        comparison_img: PIL Image with 3 subplots side by side
    """
    # 1. 相位彩色图
    fig1, ax1 = plt.subplots(figsize=(4, 4), dpi=50)
    im1 = ax1.imshow(phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    ax1.set_title(f'Phase Hologram (Digit {digit_label})', fontsize=10)
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    fig1.canvas.draw()
    img1 = np.frombuffer(fig1.canvas.tostring_rgb(), dtype=np.uint8)
    img1 = img1.reshape(fig1.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig1)
    
    # 2. 相位灰度16bit
    phase_16bit = ((phase + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
    fig2, ax2 = plt.subplots(figsize=(4, 4), dpi=50)
    ax2.imshow(phase_16bit, cmap='gray')
    ax2.set_title('Phase (8-bit Gray)', fontsize=10)
    ax2.axis('off')
    fig2.canvas.draw()
    img2 = np.frombuffer(fig2.canvas.tostring_rgb(), dtype=np.uint8)
    img2 = img2.reshape(fig2.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig2)
    
    # 3. 恢复的数字图像
    recovered = recover_image_from_phase(phase)
    recovered_uint8 = (recovered * 255).astype(np.uint8)
    fig3, ax3 = plt.subplots(figsize=(4, 4), dpi=50)
    ax3.imshow(recovered_uint8, cmap='gray')
    ax3.set_title('Recovered Digit', fontsize=10)
    ax3.axis('off')
    fig3.canvas.draw()
    img3 = np.frombuffer(fig3.canvas.tostring_rgb(), dtype=np.uint8)
    img3 = img3.reshape(fig3.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig3)
    
    # 并排组合
    h = img1.shape[0]
    w = img1.shape[1]
    comparison = np.hstack([img1, img2, img3])
    
    return Image.fromarray(comparison)


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
    """Save generated phases and recovered digits as separate files."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 创建子目录
    phase_dir = os.path.join(output_dir, 'phase_data')        # 原始相位数据
    recovered_dir = os.path.join(output_dir, 'recovered')      # 恢复的数字图像
    comparison_dir = os.path.join(output_dir, 'comparison')    # 对比图
    
    Path(phase_dir).mkdir(parents=True, exist_ok=True)
    Path(recovered_dir).mkdir(parents=True, exist_ok=True)
    Path(comparison_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 保存生成的相位和恢复图像...")
    
    for class_id in range(10):
        class_mask = labels == class_id
        class_samples = samples[class_mask]  # (batch_size, 1, 256, 256)
        
        for idx, sample in enumerate(class_samples):
            phase = sample[0].numpy()  # (256, 256), 范围 [-π, π]
            
            # 1. 保存原始相位 NPY
            phase_npy_path = os.path.join(phase_dir, f'phase_digit{class_id}_{idx:03d}.npy')
            np.save(phase_npy_path, phase)
            
            # 2. 恢复图像 PNG
            recovered_img = phase_to_recovered_image(phase)
            recovered_png_path = os.path.join(recovered_dir, f'digit{class_id}_{idx:03d}_recovered.png')
            recovered_img.save(recovered_png_path)
            
            # 3. 对比图 (相位彩色 + 相位灰度 + 恢复数字)
            comparison_img = create_comparison_image(phase, class_id)
            comparison_path = os.path.join(comparison_dir, f'digit{class_id}_{idx:03d}_comparison.png')
            comparison_img.save(comparison_path)
    
    print(f"  ✓ 相位数据: {phase_dir}/")
    print(f"  ✓ 恢复数字: {recovered_dir}/")
    print(f"  ✓ 对比图: {comparison_dir}/")


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
