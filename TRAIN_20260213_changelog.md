# Training 调整记录 - TRAIN_20260213

## 问题诊断

### 0710 训练 (LR=1e-4, warmup 5 epochs + cosine decay)
- Epoch 1-3: Loss 从 1.37 降至 1.00，warmup 阶段正常
- Epoch 4 (step 3637, LR~8e-5): **梯度爆炸**，gradient norm 持续 NaN/Inf
- 根因：peak LR=1e-4 过高，且 NaN 恢复机制仅监控 loss NaN（loss 始终有限），未监控 gradient norm NaN → 恢复从未触发
- 模型从 epoch 5 到 epoch 27 完全停止学习（所有 optimizer step 被跳过）

### 0711 训练 (LR=3e-5, warmup 5 epochs + cosine decay)
- 稳定性优秀：仅 4 次孤立 NaN 梯度（epoch 38, 均 consecutive:1）
- 收敛偏慢：epoch 60 时 loss≈0.77-0.80，而 0703 同期为 0.68
- 原因：5 epoch warmup 代价 + cosine 衰减过早降低 LR

### 0703 训练 (固定 LR=3e-5, 无 warmup/decay)
- 前 140 epoch 稳定，loss 持续下降至 0.337
- Epoch 140 梯度爆炸（永久 NaN），训练终止
- 说明：固定 LR=3e-5 长期可行但后期不安全

---

## 修改内容

### 1. NaN 恢复机制修复（0710 问题修复）

**问题**: `nan_count` 仅追踪 NaN loss，但实际只有 gradient norm 为 NaN → 恢复永不触发

**修复**:
- 新增 `nan_grad_count` 追踪连续 NaN gradient norm 次数
- 连续 20 次 NaN gradient norm 触发恢复（加载 best checkpoint + LR 减半）
- 恢复时同时修改 `scheduler.base_lrs` 和 `param_group['lr']`，确保 LR 减半持久生效
- 减少日志刷屏：每 50 次 NaN gradient 才打印一条警告

### 2. Gradient Clipping 收紧

- `max_norm`: 1.0 → **0.5**

### 3. LR 调度策略重设计（三阶段）

**旧策略**: warmup 5 epochs → 立即 cosine decay 245 epochs

**新策略**:
```
|-- 3 epochs --|---------- 97 epochs ----------|-------- 150 epochs --------|
   Warmup            Flat (peak LR=5e-5)          Cosine decay → 1e-6
  5e-6 → 5e-5           constant 5e-5             5e-5 → 1e-6
```

| 参数 | 旧值 (0711) | 新值 |
|------|------------|------|
| Peak LR | 3e-5 | **5e-5** |
| Warmup epochs | 5 | **3** |
| Flat epochs | 0 | **97** |
| Decay epochs | 245 | **150** |
| Decay start | epoch 5 | **epoch 100** |

**设计理由**:
- Peak LR 提高到 5e-5：加速收敛（0703 证明固定 3e-5 可训练 140 epoch，5e-5 配合 grad_clip=0.5 + NaN 恢复应可承受）
- 前 100 epoch 保持峰值 LR：给模型充分的学习动力（避免 0711 的过早衰减问题）
- 后 150 epoch cosine 衰减：防止 0703 式的后期梯度爆炸

### 4. Trainer 参数更新

- 新增 `flat_epochs` 参数传入 Trainer
- W&B config 新增 `flat_epochs` 和 `decay_epochs` 字段

---

## 预期效果

| 指标 | 0711 (旧) | 新策略 (预期) |
|------|-----------|--------------|
| Epoch 1-3 loss | 1.34→1.30 (warmup慢) | ~1.2 (warmup更快) |
| Epoch 10 loss | 0.94 | ~0.87 (接近0703) |
| Epoch 50 loss | 0.77 | ~0.65 (超越0703) |
| Epoch 140 | ~0.55 (估) | ~0.30 (目标) |
| 稳定性 | NaN恢复未触发 | NaN恢复作为安全网 |

---

## 运行命令

```bash
nohup python3 train.py > train_20260713.log 2>&1 &
```

---

## 后续改进方向：FFT 域辅助 Loss

### 问题
模型虽然 loss 低，但生成的相位全息图经 FFT 恢复后无法产生可辨识数字。
Loss 和感知质量不对齐（loss-perception gap）。

### 方案
在噪声预测 loss 基础上，增加 FFT 域辅助 loss：

```python
# 现有 loss
noise_pred = model(x_t, t, label)
loss_noise = MSE(noise_pred, real_noise)

# 新增 FFT loss
x_0_pred = (x_t - sqrt(1-alpha_t) * noise_pred) / sqrt(alpha_t)
x_0_pred = clamp(x_0_pred, -1, 1)
phase_pred = x_0_pred * π
phase_real = x_0 * π
amp_pred = |FFT(exp(i * phase_pred))|   # 模型生成的"数字图像"
amp_real = |FFT(exp(i * phase_real))|   # 真实的"数字图像"
amp_pred = amp_pred / (amp_pred.max() + 1e-8)
amp_real = amp_real / (amp_real.max() + 1e-8)
loss_fft = MSE(amp_pred, amp_real)

# 总 loss（仅在低 t 时启用 FFT loss）
if t < 500:
    total_loss = loss_noise + λ * loss_fft   # λ ≈ 0.1
else:
    total_loss = loss_noise
```

---

## 0703 vs 0715 训练差异深度分析

### 配置对比

| | 0703 | 0715 |
|---|---|---|
| GPU | **1× A40** | 3× A40 DDP |
| batch_size/GPU | 64 | 48 |
| gradient_accumulation | 1 | **2** |
| 有效 batch size | **64** | 48×3×2 = **288** |
| batches/epoch/GPU | **938** | 417 |
| **optimizer steps/epoch** | **938** | 417/2 = **208** |
| LR | 3e-5 **固定** | 3e-5 warmup+flat+cosine |
| 总 epochs | 156 (crash) | 250 |
| Best loss epoch | 140 | 223 |
| Best loss | **0.337** | 0.392 |
| **总 optimizer steps** | 140×938 ≈ **131K** | 223×208 ≈ **46K** |
| 推理生成效果 | 模糊可辨 | 黑乎乎一片 |

### 核心发现：优化器步数差 2.8 倍

0703 到最佳 loss 时执行了约 **131,000** 次参数更新，而 0715 只执行了约 **46,000** 次 — 差了 **2.8 倍**。

原因链路：
```
3 卡 DDP → 每 GPU 数据量 /3 → batches/epoch 从 938 降到 417
gradient_accumulation=2 → optimizer steps/epoch 再 /2 → 仅 208 步
总步数 = 208 × 223 = 46K vs 0703 的 938 × 140 = 131K
```

### 三个加剧因素

**1. 有效学习率偏低**
- 有效 batch 从 64 增到 288（4.5×），按线性缩放规则 LR 应提高到 3e-5 × √4.5 ≈ 6.4e-5
- 但实际 LR 仍为 3e-5 → 相对学习率不足

**2. Cosine decay 进一步降低后期 LR**
- 0703: 140 epoch 全程 LR = 3e-5
- 0715: epoch 100 起 LR 开始衰减，epoch 223 时 LR ≈ 2e-6（已衰减 15×）
- 后期学习动力不足，无法继续细化相位结构

**3. 梯度噪声减少**
- 0703 batch=64 → 梯度噪声大 → 有助于探索 loss landscape、逃离局部最优
- 0715 有效 batch=288 → 梯度更平滑 → 可能陷入更平坦的局部最优，丢失高频相位细节

### 结论

0715 虽然训练稳定（零 NaN），但 DDP 配置导致实际优化步数严重不足，加上 cosine decay 后期 LR 过低，模型没有足够的学习动力来掌握正确的相位排列结构。

### 建议方案

| 方案 | 做法 | 预期效果 |
|------|------|----------|
| **A. 去掉 grad_accum** | grad_accum=2→1, steps/epoch=417 | 步数翻倍，接近 0703 的一半 |
| **B. 延长训练** | 目标 630 epoch（417步×630≈131K 总步数） | 对齐 0703 总步数 |
| **C. 提高 LR** | peak_lr=3e-5→5e-5（配合 NaN recovery） | 补偿大 batch 的学习率不足 |
| **D. 综合方案** | grad_accum=1 + LR=4e-5 + 350 epoch | 平衡步数和 LR |
