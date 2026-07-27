# 0718 训练记录 — 4×A40 DDP 训练

## 基本信息

| 项目 | 值 |
|------|-----|
| 训练日期 | 2026-07-18 |
| Git Commit | `8a1be2e` (modify for 4 A40 GPU) |
| 日志文件 | `train_20260718.log` |
| W&B Run | `conditional_dit_phase_hologram_ddp` |
| GPU | 4× NVIDIA A40 (47.74 GB each, 总显存 203.73 GB) |
| 启动命令 | `tmux new -s train -d "CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py --config 4_GPU_256x256 > train_20260718.log 2>&1"` |
| 开始时间 | 2026-07-18 20:20:08 |
| 当前状态 | 已暂停 (epoch 316/550, ~57.5%) |
| 暂停时间 | 2026-07-21 |

---

## 模型配置

| 参数 | 值 |
|------|-----|
| 模型类型 | ConditionalDiT-B |
| 模型参数量 | 114,924,928 (114.9M) |
| img_size | 256 |
| patch_size | 8 |
| in_channels | 1 |
| hidden_dim | 768 |
| num_layers | 12 |
| num_heads | 12 |
| time_dim | 256 |
| num_classes | 10 |
| mlp_ratio | 4 |

---

## 训练超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| num_epochs | 550 | 计划训练 550 epoch |
| batch_size_per_gpu | 48 | 每 GPU 48 个样本 |
| gradient_accumulation_steps | 1 | 无梯度累积 |
| **effective_batch_size** | **192** | 48 × 4 GPUs |
| learning_rate | 3e-5 | 峰值学习率 |
| weight_decay | 0.01 | 与 0703 一致 |
| grad_clip_max_norm | 1.0 | 与 0703 一致 |
| timesteps | 1000 | DDPM 扩散步数 |
| schedule_type | cosine | 噪声调度类型 |
| sample_interval | 20 | 每 20 epoch 采样一次 |

### DDP 数据分配

```
60,000 样本 / 4 GPUs = 15,000 样本/GPU
15,000 / batch_size 48 = 312.5 → 313 batches/GPU/epoch
```

---

## 学习率调整策略

### 三阶段 LR 调度

```
|-- 5 epochs --|---------- 350 epochs ----------|-------- 195 epochs --------|
   Warmup            Flat (peak LR=3e-5)          Cosine decay → 1e-6
  3e-6 → 3e-5           constant 3e-5             3e-5 → 1e-6
```

| 阶段 | Epochs | Steps | LR 范围 |
|------|--------|-------|---------|
| Warmup | 1-5 | 0 - 1,565 | 3e-6 → 3e-5 |
| Flat | 6-355 | 1,565 - 111,315 | 恒定 3e-5 |
| Cosine Decay | 356-550 | 111,315 - 172,150 | 3e-5 → 1e-6 |

---

## 数据集

| 项目 | 值 |
|------|-----|
| 训练集 | 60,000 样本 (6 × 10,000 NPZ 分块) |
| 测试集 | 10,000 样本 (1 × NPZ 分块) |
| 图像尺寸 | 256 × 256 |
| 数据类型 | phase hologram |
| 相位范围 | [-π, π] ≈ [-3.142, 3.142] |
| 归一化后 | [-1, 1] |
| 标签范围 | 0-9 (10 类数字) |

---

## 训练过程

### Loss 变化

| Epoch | Average Loss | 阶段 |
|-------|-------------|------|
| 1 | 1.205 | warmup |
| 10 | 0.939 | |
| 20 | 0.741 | |
| 40 | 0.803 | 波动中下降 |
| 60 | 0.713 | |
| 80 | 0.563 | |
| 100 | 0.509 | |
| 120 | 0.383 | |
| 140 | 0.227 | 快速收敛期 |
| 160 | 0.211 | |
| 180 | 0.201 | |
| 200 | 0.194 | |
| 220 | 0.188 | |
| 221 | 0.187 | |
| 232 | 0.186 | |
| 240 | 0.187 | |
| 250 | 0.186 | |
| 260 | 0.185 | |
| 270 | 0.183 | |
| 280 | 0.184 | |
| 290 | 0.183 | |
| 300 | 0.179 | |
| **306** | **0.177** | **当前最佳** |
| 316 | 0.179 | 暂停前最后完整 epoch |

### 训练曲线特征

```
Loss
1.2 |*
    | *
1.0 |  **
    |    ***
0.8 |       ****
    |           *****
0.6 |                ******
    |                    ****
0.4 |                        ***
    |                           ****
0.2 |                               ****  ← 当前 (0.187)
    |                                    ****
0.0 +---|---|---|---|---|---|---|---|---|--→ Epoch
    0   50  100 150 200 250 300 350 400 550
```

### 关键节点

| 事件 | Epoch | Step | 说明 |
|------|-------|------|------|
| 训练开始 | 1 | 0 | Loss = 1.205 |
| Loss < 0.5 | ~100 | ~31,300 | 进入快速收敛期 |
| Loss < 0.2 | ~180 | ~56,340 | 收敛加速 |
| **最佳 Loss** | **306** | **~95,778** | **0.176916** |
| 当前进度 | 316 | ~98,900 | 已暂停 |

### 训练统计

| 指标 | 值 |
|------|-----|
| 当前 Epoch | 316 / 550 (~57.5%) |
| 总 Optimizer Steps | ~98,900 |
| Batches per Epoch | 313 |
| Optimizer Steps per Epoch | 313 |
| NaN 事件 | **0** |
| NaN 恢复触发 | **0** |

---

## 训练效果

### 与 0703 对比

| | 0703 | 0718 | 改进 |
|---|------|------|------|
| Best Loss | 0.337 (epoch 140) | **0.177 (epoch 306)** | **降低 47%** |
| 训练稳定性 | epoch 157 梯度爆炸 | **316 epoch 零 NaN** | 显著提升 |
| 总步数 | ~146K | ~99K (已暂停) | - |
| 目标总步数 | - | 172K | 多 18% |

### 关键改进

1. **Loss 大幅下降** — 从 0.337 降至 0.177，降低 47%
2. **训练稳定** — 316 epoch 零 NaN，NaN recovery 未触发
3. **收敛速度快** — 比 0703 同期低很多

---

## 推理效果

### 训练中采样

| 项目 | 值 |
|------|-----|
| 采样方法 | DDPM (Denoising Diffusion Probabilistic Models) |
| 采样步数 | 1000 步 |
| 采样频率 | 每 20 epoch |
| 输出内容 | 10 个数字 (0-9) 的 FFT 恢复图像 |
| 输出文件 | `outputs/recovered_digits_epoch_{epoch}.png` |

### 已生成采样

- epoch 20: `outputs/recovered_digits_epoch_20.png`
- epoch 40: `outputs/recovered_digits_epoch_40.png`
- epoch 60: `outputs/recovered_digits_epoch_60.png`
- epoch 80: `outputs/recovered_digits_epoch_80.png`
- epoch 100: `outputs/recovered_digits_epoch_100.png`
- epoch 120: `outputs/recovered_digits_epoch_120.png`
- epoch 140: `outputs/recovered_digits_epoch_140.png`
- epoch 160: `outputs/recovered_digits_epoch_160.png`
- epoch 180: `outputs/recovered_digits_epoch_180.png`
- epoch 200: `outputs/recovered_digits_epoch_200.png`
- epoch 220: `outputs/recovered_digits_epoch_220.png`
- epoch 232: `outputs/recovered_digits_epoch_232.png`
- epoch 240: `outputs/recovered_digits_epoch_240.png`
- epoch 260: `outputs/recovered_digits_epoch_260.png`
- epoch 280: `outputs/recovered_digits_epoch_280.png`
- epoch 300: `outputs/recovered_digits_epoch_300.png`

---

## 训练调整与备注

### flat_epochs 调整

- 训练过程中已将 `config/4_GPU_256x256.yaml` 中的 `flat_epochs` 从 **350** 调整为 **200**。
- 当前日志 (`train_20260718.log`) 覆盖 epoch 1~316，期间 LR 始终保持在 `3.00e-05`，说明本次运行仍基于旧的 `flat_epochs=350` 调度（flat 阶段应持续至 epoch 355）。
- 调整后新的三阶段调度为：
  - Warmup：5 epochs（3e-6 → 3e-5）
  - Flat：200 epochs（恒定 3e-5）
  - Cosine decay：345 epochs（3e-5 → 1e-6），从 epoch 206 开始
- 新的 `flat_epochs=200` 配置将在 **暂停后重新 resume 训练时生效**。

### 暂停原因

- 本次训练已跑完 316 个 epoch，loss 进入 0.177~0.180 的平台期。
- 为应用新的 `flat_epochs=200` 配置并继续 cosine decay 阶段，计划于 2026-07-21 暂停当前任务，resume 后继续训练。

---

## 参考文件

- 训练日志：`train_20260718.log`
- 训练代码：`train.py`
- 配置文件：`config/4_GPU_256x256.yaml`
- 模型代码：`dit_model.py`
- 扩散代码：`diffusion.py`
- 采样工具：`sample_utils.py`
- 推理脚本：`inference.py`

---

## 后续版本对比

| 版本 | 日期 | GPU | Epochs | 总步数 | Best Loss | 关键改动 |
|------|------|-----|--------|--------|-----------|---------|
| 0703 | 2026-07-03 | 1×A40 | 157 (crash) | ~146K | 0.337 | 基线，固定 LR |
| 0710 | 2026-07-10 | 1×A40 | ? | ? | ? | LR=1e-4 → 爆炸 |
| 0711 | 2026-07-11 | 1×A40 | ? | ? | ? | LR=3e-5 + warmup |
| 0713 | 2026-07-13 | 1×A40 | ? | ? | ? | 三阶段 LR + NaN 恢复 |
| 0715 | 2026-07-15 | 3×A40 | 232 | ~46K | 0.392 | DDP + grad_accum=2 |
| **0718** | **2026-07-18** | **4×A40** | **316/550 (暂停)** | **~99K** | **0.177** | **对齐 0703 配置** |
