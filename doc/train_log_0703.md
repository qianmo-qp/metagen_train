# 0703 训练记录 — Conditional DiT Phase Hologram

## 基本信息

| 项目 | 值 |
|------|-----|
| 训练日期 | 2026-07-03 |
| Git Commit | `dbafaa2` (Increase batch_size 32→64) |
| 日志文件 | `train_20260703.log` |
| W&B Run | `conditional_dit_phase_hologram` |
| GPU | 1× NVIDIA A40 (47.74 GB) |
| 启动命令 | `nohup python3 train.py > train_20260703.log 2>&1 &` |
| 开始时间 | 2026-07-03 13:09:10 |
| 结束时间 | Epoch 157 (NaN 崩溃) |

---

## 模型配置

| 参数 | 值 |
|------|-----|
| 模型类型 | ConditionalDiT |
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
| num_epochs | 400 | 计划训练 400 epoch |
| batch_size | 64 | 每 batch 64 个样本 |
| gradient_accumulation_steps | 2 | 梯度累积 2 步 |
| **effective_batch_size** | **128** | 64 × 2 = 128 |
| learning_rate | 3e-5 | AdamW 固定学习率 |
| weight_decay | 0.01 | AdamW 默认值 |
| optimizer | AdamW | 无额外配置 |
| grad_clip | 1.0 | max_norm=1.0 |
| timesteps | 1000 | DDPM 扩散步数 |
| schedule_type | cosine | 噪声调度类型 |
| sample_interval | 20 | 每 20 epoch 采样一次 |

---

## LR 调度

```python
# CosineAnnealingLR 配置
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs * 1000,  # = 400,000 步
    eta_min=1e-7
)
```

**实际效果**：
- T_max 设为 400,000 步，但实际只跑了 ~146,000 步（36.5% 进度）
- LR 从 3e-5 缓慢衰减到约 2.4e-5（几乎恒定）
- **本质上接近固定学习率训练**

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
| 1 | 1.008 | 初始快速下降 |
| 5 | 0.938 | |
| 10 | 0.870 | |
| 20 | 0.834 | |
| 40 | 0.780 | 稳步下降 |
| 60 | 0.679 | |
| 80 | 0.574 | |
| 100 | 0.409 | 进入快速收敛期 |
| 120 | 0.359 | |
| **140** | **0.337** | **最佳 Loss** |
| 148 | 0.377 | 开始波动 |
| 150 | 0.357 | |
| 155 | 0.809 | Loss 急剧上升 |
| 156 | 0.947 | 梯度爆炸前兆 |
| 157 | NaN | **训练崩溃** |

### 训练曲线特征

```
Loss
1.0 |*
    | *
0.8 |  **
    |    ***
0.6 |       ****
    |           *****
0.4 |                ******
    |                    ****  ← 最佳点 (0.337)
0.2 |                        ***
    |                           ↑ 梯度爆炸
0.0 +---|---|---|---|---|---|---|---|--→ Epoch
    0   20  40  60  80 100 120 140 157
```

### 关键节点

| 事件 | Epoch | Step | 说明 |
|------|-------|------|------|
| 训练开始 | 1 | 0 | Loss = 1.008 |
| Loss < 0.5 | ~100 | ~93,800 | 进入快速收敛期 |
| **最佳 Loss** | **140** | **~131,320** | **0.337** |
| Loss 回升 | 155 | ~144,600 | 0.809，开始不稳定 |
| NaN 崩溃 | 157 | ~146,428 | 训练终止 |

---

## 训练统计

| 指标 | 值 |
|------|-----|
| 总 Epoch 数 | 157 (计划 400) |
| 总 Optimizer Steps | ~146,000 |
| Batches per Epoch | 938 |
| Optimizer Steps per Epoch | 469 (938 / 2) |
| 总训练时长 | ~约 14 小时 |
| 最终状态 | NaN 梯度爆炸，训练终止 |

---

## Checkpoint 策略 (0703 版本)

```python
# 每个 epoch 都保存完整 checkpoint
checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
torch.save(checkpoint, checkpoint_path)

# 同时维护 best_model.pt
if loss < best_loss:
    best_loss = loss
    torch.save(checkpoint, 'best_model.pt')
```

---

## 推理效果

| 模型 | Epoch | Loss | FFT 恢复效果 |
|------|-------|------|-------------|
| best_model.pt | 140 | 0.337 | **模糊可辨数字** |
| checkpoint_epoch_154.pt | 154 | 0.528 | 质量下降 |

> 注：虽然 loss 最低为 0.337，但推理生成的数字图像只是"模糊可辨"，并非清晰。这说明 loss 和感知质量之间存在差距。

---

## 后续改进方向

### 问题诊断

1. **梯度爆炸**：固定 LR=3e-5 在后期导致梯度爆炸
2. **无 NaN 恢复机制**：0703 代码没有 NaN 检测和恢复逻辑
3. **Loss-感知不对齐**：低 loss 不等于好的生成质量

### 后续训练版本

| 版本 | 日期 | 主要改动 |
|------|------|---------|
| 0710 | 2026-07-10 | LR=1e-4 → 梯度爆炸 |
| 0711 | 2026-07-11 | LR=3e-5 + warmup + cosine |
| 0713 | 2026-07-13 | 三阶段 LR + NaN 恢复 |
| 0715 | 2026-07-15 | 3×A40 DDP |
| 0718 | 2026-07-18 | 4×A40 DDP + 对齐 0703 配置 |

---

## 参考文件

- 训练日志：`train_20260703.log`
- Git 历史：`git show dbafaa2`
- 模型代码：`dit_model.py`
- 扩散代码：`diffusion.py`
