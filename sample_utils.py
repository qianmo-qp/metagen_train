"""
Shared sampling utilities for both training validation and standalone inference.

- generate_samples_ddpm: DDPM 1000-step sampling, model output → phase [-π, π]
- recover_image_from_phase: phase [-π, π] → grayscale digit image [0, 1]
- make_digit_grid: render a grid of recovered digit images as a matplotlib figure
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageEnhance


def generate_samples_ddpm(model, diffusion, device, class_labels):
    """
    Generate phase hologram samples via DDPM (full 1000-step).

    Args:
        model:        Trained ConditionalDiT (or DDP-wrapped)
        diffusion:    DiffusionSchedule object (tensors already on device)
        device:       torch device
        class_labels: (N,) LongTensor of digit class IDs

    Returns:
        phase_samples: (N, 1, img_size, img_size) numpy float32 array in [-π, π]
    """
    raw_model = model.module if hasattr(model, 'module') else model
    raw_model.eval()

    batch_size = len(class_labels)
    class_labels = class_labels.to(device)

    # 从模型动态读取输入分辨率（兼容 64×64 / 256×256 等配置）
    img_size = getattr(raw_model, 'img_size', 256)
    in_channels = getattr(raw_model, 'in_channels', 1)

    with torch.no_grad():
        x_t = torch.randn(batch_size, in_channels, img_size, img_size, device=device)
        for t in reversed(range(diffusion.timesteps)):
            t_tensor = torch.full((batch_size,), t, dtype=torch.long, device=device)
            x_t = diffusion.p_sample(raw_model, x_t, t_tensor, class_labels,
                                     clip_denoised=True)

    # Model output is in [-1, 1]; convert back to real phase range [-π, π]
    # (Training normalizes: phase_norm = phase / π)
    phase_samples = x_t.cpu().numpy() * np.pi
    return phase_samples.astype(np.float32)


def recover_image_from_phase(phase):
    """
    Recover a digit grayscale image from its phase hologram via FFT.

    Physical basis: GS algorithm ensures |FFT(exp(i·φ))| ≈ target_image,
    so the amplitude of the far-field (Fraunhofer) diffraction pattern directly
    reconstructs the original digit — no log/sqrt post-processing needed.

    Args:
        phase: (H, W) array in [-π, π]

    Returns:
        recovered: (H, W) float32 array in [0, 1]
    """
    u1 = np.exp(1j * phase.astype(np.float64))
    u2 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u1)))
    amplitude = np.abs(u2)
    recovered = amplitude / (amplitude.max() + 1e-8)
    return recovered.astype(np.float32)


def enhance_recovered_image(img_array, denoise=0, black_point=0.0, contrast=1.0,
                            gamma=1.0, sharpen=0.0):
    """
    Post-process a recovered digit image to improve visual clarity.

    Applied in the order: denoise → black_point → contrast → gamma → sharpen.
    - Denoise removes salt-and-pepper noise (isolated dark/bright spots).
    - Black point suppresses background: pixels below this threshold are pushed
      to black, and the remaining range is rescaled to [0, 1]. This effectively
      removes the gray haze and white speckles outside the digit contour.
    - Contrast/gamma enhance the cleaned signal.
    - Sharpen restores edge definition.

    Args:
        img_array:    (H, W) float32 array in [0, 1]
        denoise:      median filter kernel size (0 = disabled, 3 or 5 recommended).
        black_point:  intensity threshold in [0, 1]. Pixels below this are set to 0;
                      the rest is rescaled to fill [0, 1]. Try 0.15~0.30.
        contrast:     contrast enhancement factor (1.0 = no change, >1 increases)
        gamma:        gamma correction (1.0 = no change, <1 brightens mid-tones)
        sharpen:      sharpening strength (0.0 = none, 1.0 = standard unsharp mask)

    Returns:
        enhanced: (H, W) float32 array in [0, 1]
    """
    img = Image.fromarray((np.clip(img_array, 0, 1) * 255).astype(np.uint8))

    # 1. Denoise: median filter to remove salt-and-pepper noise
    if denoise and int(denoise) >= 3:
        img = img.filter(ImageFilter.MedianFilter(size=int(denoise)))

    # 2. Black point: suppress background noise (white speckles outside digit)
    if black_point > 0:
        arr = np.array(img).astype(np.float32) / 255.0
        arr = np.clip((arr - float(black_point)) / (1.0 - float(black_point)), 0, 1)
        img = Image.fromarray((arr * 255).astype(np.uint8))

    # 3. Contrast enhancement
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(float(contrast))

    # 4. Gamma correction (brighten/darken mid-tones)
    if gamma != 1.0:
        arr = np.array(img).astype(np.float32) / 255.0
        arr = np.clip(arr ** float(gamma), 0, 1)
        img = Image.fromarray((arr * 255).astype(np.uint8))

    # 5. Sharpening (unsharp mask)
    if sharpen > 0:
        percent = int(150 * float(sharpen))
        radius = 2
        threshold = 3
        img = img.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))

    return np.array(img).astype(np.float32) / 255.0


def make_digit_grid(phase_samples, class_labels, title='Recovered Digits', ncols=5,
                    resolution=None, denoise=0, black_point=0.0, contrast=1.0,
                    gamma=1.0, sharpen=0.0):
    """
    Render a grid of recovered digit images.

    Args:
        phase_samples: (N, 1, H, W) numpy array in [-π, π]
        class_labels:  list/array of N digit class IDs
        title:         figure suptitle
        ncols:         columns in the grid
        resolution:    output image resolution (e.g., 28 for 28×28, None for original H×W)
        denoise:       median filter kernel size for noise removal (0=off, 3 or 5 recommended)
        black_point:   background suppression threshold (0=off, try 0.15~0.30)
        contrast:      post-processing contrast factor (default 1.0)
        gamma:         post-processing gamma (default 1.0; try 0.6~0.8 to whiten digits)
        sharpen:       post-processing sharpen strength (default 0.0; try 1.0~2.0)

    Returns:
        fig: matplotlib Figure (caller should call plt.close(fig) after saving)
    """
    n = len(class_labels)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    axes = np.array(axes).flatten()

    for i, digit in enumerate(class_labels):
        phase = phase_samples[i, 0]           # (H, W)
        recovered = recover_image_from_phase(phase)

        # Resize if resolution specified
        if resolution is not None:
            img = Image.fromarray((recovered * 255).astype(np.uint8))
            img = img.resize((resolution, resolution), Image.Resampling.LANCZOS)
            recovered = np.array(img).astype(np.float32) / 255.0

        # Apply optional post-processing to improve clarity
        if denoise or black_point > 0 or contrast != 1.0 or gamma != 1.0 or sharpen > 0:
            recovered = enhance_recovered_image(
                recovered, denoise=denoise, black_point=black_point,
                contrast=contrast, gamma=gamma, sharpen=sharpen
            )

        axes[i].imshow(recovered, cmap='gray')
        axes[i].set_title(f'Digit {digit}', fontsize=11)
        axes[i].axis('off')

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].axis('off')

    plt.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()
    return fig
