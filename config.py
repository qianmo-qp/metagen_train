"""
Training configuration loader.

Reads YAML config files from the `config/` directory.
Config file names include resolution suffix (e.g. `_256x256`, `_64x64`) to
indicate which input resolution they target.

Each config carries:
    - data_dir : phase NPZ data root (contains train/ and test/ subdirs)
    - model.img_size / model.patch_size : input resolution & patchification
    - training hyperparameters (LR schedule, batch size, etc.)

Usage in train.py:
    from config import TRAIN_CONFIGS
    cfg = TRAIN_CONFIGS['4_GPU_256x256']

Available configs:
    - 4_GPU_256x256  : 4×A40 DDP training (0718), 256×256 input, data/minst_phase
    - 1_GPU_256x256  : 1×A40 single-GPU baseline, 256×256 input, data/minst_phase
    - 4_GPU_64x64    : 4×A40 DDP training, 64×64 input, data/minst_phase_64
    - 1_GPU_64x64    : 1×A40 single-GPU, 64×64 input, data/minst_phase_64
"""

import os
import yaml
from pathlib import Path

# Directory containing YAML config files
_CONFIG_DIR = Path(__file__).parent / 'config'


def _load_config(name: str) -> dict:
    """Load a YAML config file by name."""
    yaml_path = _CONFIG_DIR / f'{name}.yaml'
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Config '{name}' not found at {yaml_path}\n"
            f"Available configs: {list_available_configs()}"
        )
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def list_available_configs() -> list:
    """List all available config names."""
    return [
        p.stem for p in _CONFIG_DIR.glob('*.yaml')
        if p.is_file()
    ]


# Lazy-loaded config dict
TRAIN_CONFIGS = {
    name: _load_config(name)
    for name in list_available_configs()
}
