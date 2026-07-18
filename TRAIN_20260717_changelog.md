# Training Changelog — 0715

## 背景

0713 训练（peak LR=5e-5 + NaN recovery）失败：
- Epoch 13 即触发 NaN 爆炸，LR 减半至 2.5e-5
- Epoch 55 再次爆炸，LR 再减半至 1.25e-5
- 有效 LR 被迫坍缩，最终收敛停滞在 loss ≈ 0.72
- 原计划 "LR=5e-5 + NaN 恢复兜底" 策略不可行

## 本轮修改（方案 A：稳健版）

### 超参数调整

| 参数 | 0713 | 0715 | 说明 |
|------|------|------|------|
| peak_lr | 5e-5 | **3e-5** | 0703 验证稳定 140 epoch |
| warmup_epochs | 3 | **5** | 更平缓进入，减少早期不稳定 |
| flat_epochs | 97 | **95** | warmup+flat 仍=100 epoch |
| weight_decay | 0.01 | **0.03** | 更强正则化，抑制权重膨胀 |
| NaN recovery | LR × 0.5 | **LR × 0.7** | 避免 LR 快速坍缩 |
| grad_clip | 0.5 | 0.5 | 不变 |

### LR Schedule（3 阶段）

```
Warmup:   5 epoch   → LR 从 3e-6 线性升到 3e-5
Flat:     95 epoch  → LR 保持 3e-5 不变
Cosine:   150 epoch → LR 从 3e-5 衰减到 1e-6
```

### 硬件配置（3× A40 40GB）

| 参数 | 值 |
|------|-----|
| 训练卡 | GPU 1, 2, 3（A40 40GB × 3） |
| batch_size_per_gpu | 48 |
| gradient_accumulation | 2 |
| 有效 batch size | 48 × 3 × 2 = **288** |

### Checkpoint 策略

仅保留两个文件，节省磁盘空间：
- `checkpoint_latest.pt` — 每 epoch 覆盖，断点续训用
- `best_model.pt` — loss 创新低时覆盖，NaN recovery 和推理用

### 其他修复

- `sample_utils.py`：抽取 DDPM 采样 + FFT 恢复为共享工具类
- `inference.py`：修复 ×π 缺失 bug，恢复逻辑对齐 GS 正向模型（amplitude / max）
- `diffusion.py`：DDIM timestep schedule 修复（末尾追加 -1，确保完全去噪）

## 启动命令

```bash
cd /mnt/model_data/qp/metagen_train

CUDA_VISIBLE_DEVICES=1,2,3 nohup torchrun --nproc_per_node=3 train.py > train_20260715.log 2>&1 &
```

## 预期行为

- 前 100 epoch：LR ≈ 3e-5，行为接近 0703，应能复现 0703 收敛曲线
- Epoch 100 后：cosine decay 缓慢降低 LR，避免 0703 在 epoch 140 的爆炸
- 若触发 NaN recovery：LR 仅降 30%（3e-5 → 2.1e-5），不会像 0713 那样快速坍缩
- 总耗时预估：250 epoch × 3 A40，比单卡快 ~3 倍
