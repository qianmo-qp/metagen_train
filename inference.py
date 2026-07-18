"""
Inference script: Generate phase holograms and recover digit images via FFT.

Usage:
    python inference.py --checkpoint checkpoints/best_model.pt
    python inference.py --checkpoint checkpoints/best/checkpoint_epoch_140.pt --digits 0 1 2 3 4 5 6 7 8 9
"""

import os
import re
import torch
import argparse
from pathlib import Path

from dit_model import ConditionalDiT
from diffusion import create_diffusion
from sample_utils import generate_samples_ddpm, make_digit_grid


def load_checkpoint(checkpoint_path, model, device):
    """Load model from checkpoint, handling DDP state_dict prefix."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    if isinstance(state_dict, dict):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    epoch = checkpoint.get('epoch', '?')
    loss = checkpoint.get('loss', '?')
    print(f"✅ Loaded checkpoint: {checkpoint_path} (epoch={epoch}, loss={loss})")
    return epoch


def main():
    parser = argparse.ArgumentParser(description='Generate phase holograms and recover digits')
    parser.add_argument('--checkpoint', default='checkpoints/best_model.pt',
                        help='Path to checkpoint')
    parser.add_argument('--digits', nargs='+', type=int, default=list(range(1, 10)),
                        help='Digits to generate (default: 1-9)')
    parser.add_argument('--output-dir', default='outputs', help='Output directory')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Method: DDPM, Steps: 1000")

    # Build model
    model = ConditionalDiT(
        img_size=256, patch_size=8, in_channels=1,
        hidden_dim=768, num_heads=12, num_layers=12,
        time_dim=256, num_classes=10, mlp_ratio=4,
    ).to(device)

    epoch = load_checkpoint(args.checkpoint, model, device)
    model.eval()

    # Diffusion schedule
    diffusion = create_diffusion(timesteps=1000, schedule_type='cosine')
    for attr_name in dir(diffusion):
        attr = getattr(diffusion, attr_name)
        if isinstance(attr, torch.Tensor):
            setattr(diffusion, attr_name, attr.to(device))

    # Generate samples
    print(f"\n🎨 Generating digits {args.digits}...")
    class_labels = torch.tensor(args.digits, dtype=torch.long, device=device)
    phase_samples = generate_samples_ddpm(model, diffusion, device, class_labels)
    print(f"  Phase range: [{phase_samples.min():.3f}, {phase_samples.max():.3f}]")

    # Render grid
    print("🔬 Recovering digits from phase holograms...")
    ncols = 3 if len(args.digits) <= 9 else 5
    title = f'Epoch {epoch} — Recovered Digits (DDPM 1000 steps)'
    fig = make_digit_grid(phase_samples, args.digits, title=title, ncols=ncols)

    # Save with descriptive filename
    epoch_match = re.search(r'epoch_(\d+)', args.checkpoint)
    epoch_str = epoch_match.group(1) if epoch_match else str(epoch)
    output_filename = f'recovered_digits_ddpm_1000_{epoch_str}.png'
    output_path = os.path.join(args.output_dir, output_filename)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    import matplotlib.pyplot as plt
    plt.close(fig)

    print(f"\n✅ Saved: {output_path}")


if __name__ == '__main__':
    main()
