"""
Compare GS-projection guidance strengths (λ_max) on the same checkpoint.

Loads the model once, then for each λ in --scales generates digits 0-9 with
DDPM 1000-step sampling (identical seed → identical initial noise), recovers
digit images, and renders one combined comparison figure (rows = λ values).

Usage:
    python compare_guidance.py --checkpoint checkpoints/history/best_0718_4GPU.pt
"""

import os
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt

from dit_model import ConditionalDiT
from diffusion import create_diffusion
from inference import load_checkpoint
from sample_utils import (generate_samples_ddpm, generate_samples_ddpm_guided,
                          build_class_templates, recover_image_from_phase,
                          enhance_recovered_image)


def main():
    parser = argparse.ArgumentParser(description='GS-projection guidance λ comparison')
    parser.add_argument('--checkpoint', default='checkpoints/history/best_0718_4GPU.pt')
    parser.add_argument('--scales', nargs='+', type=float, default=[0.0, 0.2, 0.4, 0.6])
    parser.add_argument('--digits', nargs='+', type=int, default=list(range(10)))
    parser.add_argument('--img-size', type=int, default=256)
    parser.add_argument('--patch-size', type=int, default=8)
    parser.add_argument('--template', choices=['random', 'mean'], default='random')
    parser.add_argument('--mnist-dir', default='data/minst')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--output', default='outputs/guidance_lambda_comparison.png')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model = ConditionalDiT(
        img_size=args.img_size, patch_size=args.patch_size, in_channels=1,
        hidden_dim=768, num_heads=12, num_layers=12,
        time_dim=256, num_classes=10, mlp_ratio=4,
    ).to(device)
    epoch = load_checkpoint(args.checkpoint, model, device)
    model.eval()

    diffusion = create_diffusion(timesteps=1000, schedule_type='cosine')
    for attr_name in dir(diffusion):
        attr = getattr(diffusion, attr_name)
        if isinstance(attr, torch.Tensor):
            setattr(diffusion, attr_name, attr.to(device))

    templates = build_class_templates(args.mnist_dir, args.img_size, mode=args.template)
    class_labels = torch.tensor(args.digits, dtype=torch.long, device=device)

    # Generate for each λ with identical initial noise (same seed)
    results = {}
    for lam in args.scales:
        torch.manual_seed(args.seed)
        print(f"\n=== λ_max = {lam} ===")
        if lam > 0:
            phases = generate_samples_ddpm_guided(
                model, diffusion, device, class_labels,
                templates=templates, guidance_scale=lam)
        else:
            phases = generate_samples_ddpm(model, diffusion, device, class_labels)
        results[lam] = phases
        print(f"  done, phase range [{phases.min():.3f}, {phases.max():.3f}]")

    # Combined grid: rows = λ, cols = digits (+ optional recon MSE in title)
    nrows, ncols = len(args.scales), len(args.digits)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.8, nrows * 2.0))
    axes = np.atleast_2d(axes)

    for r, lam in enumerate(args.scales):
        for c, digit in enumerate(args.digits):
            phase = results[lam][c, 0]
            recovered = recover_image_from_phase(phase)
            recovered = enhance_recovered_image(
                recovered, denoise=3, black_point=0.05, contrast=1.5,
                gamma=0.7, sharpen=1.0)
            axes[r, c].imshow(recovered, cmap='gray')
            axes[r, c].axis('off')
            if r == 0:
                axes[r, c].set_title(f'{digit}', fontsize=11)
        axes[r, 0].set_ylabel(f'λ={lam}', fontsize=11)
        # ylabel invisible when axis off; add text instead
        axes[r, 0].text(-0.25, 0.5, f'λ={lam}', fontsize=12,
                        transform=axes[r, 0].transAxes,
                        va='center', ha='right')

    plt.suptitle(f'GS-projection guidance comparison — epoch {epoch}, '
                 f'template={args.template}, DDPM 1000 steps', fontsize=13, y=1.005)
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n✅ Saved: {args.output}")


if __name__ == '__main__':
    main()
