"""
Shared sampling utilities for both training validation and standalone inference.

- generate_samples_ddpm: DDPM 1000-step sampling, model output → phase [-π, π]
- recover_image_from_phase: phase [-π, π] → grayscale digit image [0, 1]
- enhance_recovered_image: optional post-processing (denoise/black point/contrast)
- make_digit_grid: render a grid of recovered digit images as a matplotlib figure
- torch_propagate / gs_project_phase: differentiable optical operators (torch)
- build_class_templates: per-class target amplitudes matching the GS data pipeline
- generate_samples_ddpm_guided: DDPM sampling with GS-projection guidance
"""

import os

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
        phase: (256, 256) array in [-π, π]

    Returns:
        recovered: (256, 256) float32 array in [0, 1]
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

    Args:
        img_array:    (H, W) float32 array in [0, 1]
        denoise:      median filter kernel size (0 = disabled, 3 or 5 recommended).
        black_point:  intensity threshold in [0, 1]. Pixels below this are set to 0;
                      the rest is rescaled to fill [0, 1].
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
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=percent, threshold=3))

    return np.array(img).astype(np.float32) / 255.0


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


# ---------------------------------------------------------------------------
# GS-projection guidance (inference-time only, independent of training)
# ---------------------------------------------------------------------------

def torch_propagate(phase):
    """
    Differentiable forward optical operator: phase → far-field amplitude.

    A(φ) = |FFT(exp(iφ))|, matching recover_image_from_phase (numpy version).

    Args:
        phase: (..., H, W) real tensor in [-π, π]
    Returns:
        amplitude: (..., H, W) real tensor
    """
    u1 = torch.exp(1j * phase)
    u2 = torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(u1, dim=(-2, -1))), dim=(-2, -1))
    return u2.abs()


def gs_project_phase(phase, target_amp):
    """
    One Gerchberg-Saxton measurement projection.

    Enforce |FFT(exp(iφ))| = target_amp while keeping the image-plane phase,
    then transform back and take the SLM-plane phase (unit amplitude).

    Args:
        phase:      (..., H, W) real tensor in [-π, π]
        target_amp: (..., H, W) target far-field amplitude (energy-normalized)
    Returns:
        phase_proj: (..., H, W) real tensor in [-π, π]
    """
    u1 = torch.exp(1j * phase)
    u2 = torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(u1, dim=(-2, -1))), dim=(-2, -1))
    phase_img = torch.angle(u2)
    u2_new = target_amp * torch.exp(1j * phase_img)
    u1_new = torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(u2_new, dim=(-2, -1))), dim=(-2, -1))
    return torch.angle(u1_new)


def _read_idx_images(path):
    """Read MNIST IDX image file → (N, 28, 28) uint8 array."""
    with open(path, 'rb') as f:
        header = np.frombuffer(f.read(16), dtype='>i4')
        num, rows, cols = int(header[1]), int(header[2]), int(header[3])
        data = np.frombuffer(f.read(), dtype='>u1')
    return data.reshape(num, rows, cols)


def _read_idx_labels(path):
    """Read MNIST IDX label file → (N,) uint8 array."""
    with open(path, 'rb') as f:
        f.read(8)
        data = np.frombuffer(f.read(), dtype='>u1')
    return data


def build_class_templates(mnist_dir, img_size, mode='random', seed=42):
    """
    Build per-class target amplitude templates matching the GS data pipeline:
    /255 → LANCZOS resize → contrast (x-0.5)*1.3+0.5 → clip → energy normalize.

    Args:
        mnist_dir: directory containing train-images/labels IDX files
        img_size:  output resolution (e.g. 256)
        mode:      'random' = one exemplar per class; 'mean' = class mean image
        seed:      RNG seed for 'random' mode
    Returns:
        templates: (10, img_size, img_size) float32, energy = img_size² each
    """
    images = _read_idx_images(os.path.join(mnist_dir, 'train-images.idx3-ubyte'))
    labels = _read_idx_labels(os.path.join(mnist_dir, 'train-labels.idx1-ubyte'))

    rng = np.random.RandomState(seed)
    templates = np.zeros((10, img_size, img_size), dtype=np.float32)
    for digit in range(10):
        idx = np.where(labels == digit)[0]
        if mode == 'mean':
            img28 = images[idx].astype(np.float32).mean(axis=0) / 255.0
        else:
            img28 = images[rng.choice(idx)].astype(np.float32) / 255.0

        # Same preprocessing as ds/minst_image_to_phase_process.py
        pil = Image.fromarray((img28 * 255).astype(np.uint8))
        pil = pil.resize((img_size, img_size), Image.LANCZOS)
        arr = np.array(pil).astype(np.float32) / 255.0
        arr = np.clip((arr - 0.5) * 1.3 + 0.5, 0, 1)

        # Energy normalization (same as gs_algorithm: total energy = N²)
        energy_target = float((arr ** 2).sum())
        arr = arr * np.sqrt(img_size * img_size / max(energy_target, 1e-12))
        templates[digit] = arr

    return templates


def generate_samples_ddpm_guided(model, diffusion, device, class_labels,
                                 templates, guidance_scale=0.4):
    """
    DDPM sampling with GS-projection guidance (DiffFPR-style, gradient-free).

    At each reverse step the predicted x̂₀ is nudged toward optical consistency
    with the class template amplitude via one GS projection (see
    diffusion.p_sample_guided). λ_t = guidance_scale · √ᾱ_t.

    Args:
        model:          Trained ConditionalDiT (or DDP-wrapped)
        diffusion:      DiffusionSchedule (tensors already on device)
        device:         torch device
        class_labels:   (N,) LongTensor of digit class IDs
        templates:      (10, H, W) numpy array from build_class_templates
        guidance_scale: λ_max in [0, 1]; 0 disables guidance
    Returns:
        phase_samples: (N, 1, H, W) numpy float32 array in [-π, π]
    """
    raw_model = model.module if hasattr(model, 'module') else model
    raw_model.eval()

    batch_size = len(class_labels)
    class_labels = class_labels.to(device)
    img_size = templates.shape[-1]

    # Per-sample target amplitude (N, 1, H, W)
    target_amp = torch.from_numpy(
        templates[class_labels.cpu().numpy()]).unsqueeze(1).to(device)

    with torch.no_grad():
        x_t = torch.randn(batch_size, 1, img_size, img_size, device=device)
        for t in reversed(range(diffusion.timesteps)):
            t_tensor = torch.full((batch_size,), t, dtype=torch.long, device=device)
            x_t = diffusion.p_sample_guided(
                raw_model, x_t, t_tensor, class_labels, target_amp,
                guidance_scale=guidance_scale, clip_denoised=True)

    phase_samples = x_t.cpu().numpy() * np.pi
    return phase_samples.astype(np.float32)

