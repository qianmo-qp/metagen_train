# Quick Start Guide - Conditional DiT for MNIST

## Setup (2 minutes)

```bash
# Install dependencies (if not already installed)
pip install torch torchvision numpy pillow matplotlib

# Check MNIST data
ls -lh data/minst/
# Should show 4 files: train-images, train-labels, t10k-images, t10k-labels
```

## Train (10-60 minutes depending on hardware)

```bash
python train.py
```

**Output:**
- Logs loss every 100 steps
- Saves checkpoint after each epoch to `checkpoints/epoch_N.pt`
- Saves best checkpoint to `checkpoints/best_model.pt`

**Expected loss:** MSE should decrease from ~0.2 to ~0.001-0.01 over 10 epochs

## Generate Samples (5-10 minutes)

After training completes:

```bash
python sample.py
```

**Output:**
- `outputs/generated_samples_1per_class.png` - 1 sample for each digit 0-9
- `outputs/generated_samples_10per_class.png` - 10 samples for each digit 0-9

## That's it! 🎉

### What's actually happening:

1. **Training (`train.py`)**:
   - Loads 60K MNIST images, normalizes to [-1, 1]
   - For each batch:
     - Sample random timestep t ∈ [0, 1000)
     - Add Gaussian noise to image: `x_t = √ᾱ·x_0 + √(1-ᾱ)·ε`
     - Train model to predict noise
   - Model: 6-layer Transformer + Adaptive LayerNorm conditioning

2. **Sampling (`sample.py`)**:
   - Start with random noise
   - Iteratively denoise for t = 1000 → 0
   - Model predicts which noise to remove at each step
   - Final result: generated MNIST digit

### Customization

Edit these in the Python files:

**Training hyperparams** (`train.py`, line ~163):
```python
num_epochs = 10           # More = better quality
batch_size = 128          # Smaller = less VRAM
learning_rate = 1e-4
timesteps = 1000
```

**Model size** (`train.py`, line ~175):
```python
hidden_dim=192       # Smaller = faster, larger = better quality
num_layers=6         # More = better but slower
```

**Sampling** (`sample.py`, line ~111):
```python
batch_size=1    # Samples per class
timesteps=1000  # More = better quality, slower (try 500 for speed)
```

### GPU Acceleration

The code automatically detects and uses:
- CUDA (NVIDIA GPUs)
- MPS (Apple Silicon: M1, M2, M3...)
- CPU fallback

No changes needed - it just works!

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory | Reduce `batch_size` in train.py (64 or 32) |
| Slow sampling | Reduce timesteps in sample.py to 500 or 250 |
| Poor quality | Train more epochs or increase `hidden_dim` to 256 |
| Data not found | Verify `data/minst/` has 4 .idx files |

## File Guide

- **`dit_model.py`** - Core model (Patchify → Transformer → Unpatchify)
- **`diffusion.py`** - Diffusion math (forward/reverse process, DDPM sampler)
- **`train.py`** - Training loop (loads data, trains, saves checkpoints)
- **`sample.py`** - Generation script (loads checkpoint, generates grids)
- **`data/minst/`** - MNIST dataset (train + test images & labels)

## Next Steps

After getting familiar with the basic setup:

1. **Understand the code**: Read comments in `dit_model.py` - architecture is intuitive
2. **Experiment**: Try different hyperparameters
3. **Improve sampling**: Implement DDIM sampler for 10x faster generation
4. **Add features**: Classifier-free guidance, EMA model, etc.

Good luck! 🚀
