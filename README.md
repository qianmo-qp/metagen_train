# Conditional DiT for MNIST

A clean PyTorch implementation of a Conditional Diffusion Transformer (DiT) for generating MNIST handwritten digits.

## Project Structure

```
metagen_train/
├── dit_model.py                # DiT architecture
├── diffusion.py                # Diffusion process (DDPM + DDIM)
├── train.py                    # Training loop with W&B monitoring
├── sample.py                   # Generation/sampling
├── data/minst/                 # MNIST dataset (IDX files)
├── data/minst_phase/           # Phase hologram data (generated)
├── ds/                         # Phase processing tools
│   ├── phase_extractor.py      # PNG → phase_map.npy
│   ├── phase_visualizer.py     # phase_map.npy → PNG
│   └── mnist_image_process.py  # Batch MNIST → phase conversion
├── checkpoints/                # Model checkpoints (auto-created)
├── outputs/                    # Generated samples (auto-created)
└── README_PHASE.md            # Phase processing guide
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Training

```bash
python train.py
```

- Loads MNIST (60K images)
- Trains with DDPM (sampling monitoring uses fast DDIM)
- Logs to W&B (set WANDB_API_KEY)
- Saves checkpoints to `checkpoints/`

### Sampling

```bash
python sample.py                     # DDIM (fast, 100 steps)
python sample.py --method ddpm       # DDPM (slow, 1000 steps, highest quality)
```

Output: Generated digit samples in `outputs/`

## Phase Hologram Processing

Convert MNIST to optical phase holograms:

```bash
cd ds/

# Single image extraction
python3 phase_extractor.py image.png -o image_phase.npy

# Batch MNIST processing
python3 mnist_image_process.py

# Visualization
python3 phase_visualizer.py phase_map.npy --recovered -od /tmp/output
```

See `ds/README_PHASE.md` for detailed usage.

## Model Architecture

- **Input**: 28×28 MNIST images → 4×4 patches (49 tokens)
- **Encoder**: 6-layer Transformer (192 hidden dim, 3 heads)
- **Conditioning**: Time embedding + Class embedding (0-9)
- **Output**: Noise prediction for reverse diffusion
- **Parameters**: ~3.9M (lightweight, trains fast)

## Key Features

✅ Conditional generation (class-guided)  
✅ DDPM training + DDIM fast sampling (50× speedup)  
✅ W&B monitoring integration  
✅ GPU/CPU/Mac compatible  
✅ Modular phase processing tools  
✅ Well-documented, clean code

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Image size | 28×28 |
| Patch size | 4×4 |
| Hidden dim | 192 |
| Layers | 6 |
| Attention heads | 3 |
| Timesteps | 1000 |
| Learning rate | 1e-4 |
| Batch size | 128 |
| Epochs | 400 |

## Results

After 400 epochs training:
- Loss: ~0.078
- Generated samples: Clear, recognizable digits
- Sample quality: High with DDIM (100 steps) or DDPM (1000 steps)

## Troubleshooting

**Out of memory?** → Reduce batch_size in train.py

**Slow training?** → Use GPU (set CUDA_VISIBLE_DEVICES)

**Generation quality poor?** → Train longer or use DDPM sampling

## References

- DiT: Scalable Diffusion Models with Transformers (Peebles & Xie, 2023)
- DDPM: Denoising Diffusion Probabilistic Models (Ho et al., 2020)
- DDIM: Denoising Diffusion Implicit Models (Song et al., 2021)
# Conditional DiT for MNIST

A minimalist, clean PyTorch implementation of a Conditional Diffusion Transformer (DiT) for generating MNIST handwritten digits.

## Overview

This project implements a **pixel-space Conditional DiT** with:
- **Patchified input**: 28×28 images split into 4×4 patches (49 tokens)
- **Transformer backbone**: 6 layers with 192 hidden dim, 3 heads
- **Conditioning**: Time embedding (sinusoidal + MLP) + Class embedding (0-9)
- **AdaLN integration**: Adaptive Layer Normalization for conditioning injection
- **Diffusion schedule**: Cosine/Linear schedule with DDPM sampling
- **Lightweight**: ~3.9M parameters, trains quickly on CPU/GPU/Mac

## Project Structure

```
metagen_train/
├── dit_model.py          # Core DiT architecture
├── diffusion.py          # Diffusion process & DDPM sampling
├── train.py              # Training loop
├── sample.py             # Sampling/generation script
├── data/minst/           # MNIST dataset (4 IDX files)
├── checkpoints/          # Trained model checkpoints (auto-created)
└── outputs/              # Generated samples (auto-created)
```

## Installation

```bash
# Install dependencies
pip install torch torchvision numpy pillow matplotlib

# Verify MNIST data is in place
ls data/minst/
# Should show:
# - train-images.idx3-ubyte
# - train-labels.idx1-ubyte
# - t10k-images.idx3-ubyte
# - t10k-labels.idx1-ubyte
```

## Usage

### 1. Training

```bash
python train.py
```

**Expected behavior:**
- Loads MNIST (60K training images)
- Trains for 10 epochs with batch size 128
- Saves checkpoints every epoch to `checkpoints/`
- Logs loss every 100 steps
- Runs on GPU if available, CPU otherwise

**Hyperparameters** (edit in `train.py`):
- `num_epochs`: 10
- `batch_size`: 128
- `learning_rate`: 1e-4
- `timesteps`: 1000 (diffusion steps)

**Expected runtime:**
- ~5-10 mins per epoch on GPU (M1/M2)
- ~30-60 mins per epoch on CPU

### 2. Sampling

After training, generate samples:

```bash
python sample.py
```

**What it does:**
- Loads latest checkpoint from `checkpoints/`
- Generates 1 sample per class (0-9)
- Generates 10 samples per class for better visualization
- Saves grid images to `outputs/`

**Output files:**
- `generated_samples_1per_class.png` - Quick visualization
- `generated_samples_10per_class.png` - Full visualization (100 samples)

### 3. Quick Test (no training)

```bash
python dit_model.py    # Test model forward pass
python diffusion.py    # Test diffusion process
```

## Architecture Details

### DiT (Diffusion Transformer)

```
Input (1×28×28)
    ↓
[Patchify] (4×4 patches → 49 tokens)
    ↓
[Linear Embedding] → 192-dim
    ↓
[Add Positional Embeddings] (learnable)
    ↓
[Time Embedding] (sinusoidal + MLP) → 256-dim
[Class Embedding] (one-hot + embedding) → 256-dim
[Condition Projection] → 192-dim
    ↓
[6× DiT Blocks with AdaLN]
    ├─ Self-Attention (3 heads)
    ├─ Adaptive LayerNorm (time + class)
    └─ MLP (hidden_ratio=4)
    ↓
[Final LayerNorm]
    ↓
[Output Head] (predict noise per patch)
    ↓
[Un-Patchify] (49 tokens → 1×28×28)
    ↓
Output (1×28×28 noise prediction)
```

### Diffusion Process

**Forward (q_sample):**
```
x_t = √(ᾱ_t) · x_0 + √(1 - ᾱ_t) · ε
```

**Reverse (p_sample):**
```
x_{t-1} ~ N(μ(x_t, t, c), σ_t²)
where μ is computed from the model's noise prediction
```

**Schedule:**
- Cosine schedule (default): smoother, more stable
- Linear schedule: standard DDPM

## Hyperparameter Reference

| Parameter | Value | Notes |
|-----------|-------|-------|
| Image size | 28×28 | MNIST standard |
| Patch size | 4×4 | 49 patches total |
| Hidden dim | 192 | Small for fast training |
| Num heads | 3 | 192 / 3 = 64-dim per head |
| Num layers | 6 | Lightweight transformer |
| Time dim | 256 | Sinusoidal embedding dim |
| Num classes | 10 | 0-9 digit labels |
| Timesteps | 1000 | Diffusion steps |
| Learning rate | 1e-4 | AdamW optimizer |
| Batch size | 128 | Adjust for memory |

## Expected Results

After ~10 epochs of training:
- **MSE Loss**: ~0.001 - 0.01
- **Generated samples**: Recognizable digits, mostly mode-collapsed initially
- **Quality improves with**: longer training, larger model, more timesteps in sampling

For better quality:
- Train for 20-50 epochs
- Increase `hidden_dim` to 256 or 384
- Use smaller timesteps in sampling (e.g., 250-500) with DDIM sampler (future enhancement)

## Code Style

- **Clean & readable**: No fancy tricks, standard PyTorch idioms
- **Self-contained**: Minimal dependencies (torch, torchvision, PIL, matplotlib)
- **Well-documented**: Clear docstrings and comments
- **Cohesive**: All logic in 4 files, easy to understand & modify

## Troubleshooting

**Q: Out of memory error?**
- Reduce `batch_size` in `train.py` (e.g., 64, 32)
- Reduce `hidden_dim` in model initialization (e.g., 128)

**Q: Sampling is slow?**
- Use fewer diffusion steps: edit `sample.py` timesteps to 500 or 250
- Sample on GPU if available

**Q: Poor generation quality?**
- Train longer (more epochs)
- Check checkpoint loaded correctly
- Verify MNIST data is normalized to [-1, 1] (done automatically)

**Q: CUDA out of memory during training?**
```python
# In train.py, reduce batch_size:
batch_size = 64  # or 32
```

## References

- **DiT**: Scalable Diffusion Models with Transformers (Peebles & Xie, 2023)
- **DDPM**: Denoising Diffusion Probabilistic Models (Ho et al., 2020)
- **AdaLN**: Adaptive Layer Normalization for Conditioning

## Future Enhancements

- [ ] DDIM sampler (faster sampling, ~50 steps)
- [ ] Classifier-free guidance (improved generation)
- [ ] EMA model (better quality)
- [ ] Validation loop
- [ ] TensorBoard logging
- [ ] Export to ONNX

## License

Free to use and modify for educational/research purposes.
