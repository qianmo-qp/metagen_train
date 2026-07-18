"""
Shared sampling utilities for both training validation and standalone inference.

- generate_samples_ddpm: DDPM 1000-step sampling, model output → phase [-π, π]
- recover_image_from_phase: phase [-π, π] → grayscale digit image [0, 1]
- make_digit_grid: render a grid of recovered digit images as a matplotlib figure
"""

import numpy as np
import torch
import matplotlib.pyplot as plt


def generate_samples_ddpm(model, diffusion, device, class_labels):
    """
    Generate phase hologram samples via DDPM (full 1000-step).

    Args:
        model:        Trained ConditionalDiT (or DDP-wrapped)
        diffusion:    DiffusionSchedule object (tensors already on device)
        device:       torch device
        class_labels: (N,) LongTensor of digit class IDs

    Returns:
        phase_samples: (N, 1, 256, 256) numpy float32 array in [-π, π]
    """
    raw_model = model.module if hasattr(model, 'module') else model
    raw_model.eval()

    batch_size = len(class_labels)
    class_labels = class_labels.to(device)

    with torch.no_grad():
        x_t = torch.randn(batch_size, 1, 256, 256, device=device)
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
        phase: (256, 256) array in [-π, π]

    Returns:
        recovered: (256, 256) float32 array in [0, 1]
    """
    u1 = np.exp(1j * phase.astype(np.float64))
    u2 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u1)))
    amplitude = np.abs(u2)
    recovered = amplitude / (amplitude.max() + 1e-8)
    return recovered.astype(np.float32)


def make_digit_grid(phase_samples, class_labels, title='Recovered Digits', ncols=5):
    """
    Render a grid of recovered digit images.

    Args:
        phase_samples: (N, 1, 256, 256) numpy array in [-π, π]
        class_labels:  list/array of N digit class IDs
        title:         figure suptitle
        ncols:         columns in the grid

    Returns:
        fig: matplotlib Figure (caller should call plt.close(fig) after saving)
    """
    n = len(class_labels)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    axes = np.array(axes).flatten()

    for i, digit in enumerate(class_labels):
        phase = phase_samples[i, 0]           # (256, 256)
        recovered = recover_image_from_phase(phase)
        axes[i].imshow(recovered, cmap='gray')
        axes[i].set_title(f'Digit {digit}', fontsize=11)
        axes[i].axis('off')

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].axis('off')

    plt.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()
    return fig
