# 最优训练归档 — 4_GPU_256x256_20260731_111654

> 本次训练的最优结果归档。该 run 为 256×256 相位全息图条件 DiT 的最终精修（resume fine-tune）训练。
> 训练开始时间：2026-07-31 11:16:56 ｜ 结束：2026-08-01 10:30
> W&B Run: [y76q6ajy](https://wandb.ai/qianmo-qp-zhejiang-lab/minst/runs/y76q6ajy)

## 归档内容

| 文件 | 说明 |
|---|---|
| `best_model.pt` | **最终最优模型权重**（epoch 389，loss 0.1878，含 EMA 权重）1.8GB |
| `recovered_digits_ddpm_1000_389_guided_0.6.png` | 推理结果图（DDPM 1000 步 + GS 投影引导 λ=0.6，EMA 权重，seed 42，数字 0-9） |
| `samples/` | 训练期间每 20 epoch 的采样图（epoch 300/320/340/360/380/400） |
| `train_0731_ft.log` | 精修阶段完整训练日志（epoch 285→400） |
| `wandb_run/` | 本次 run 的 W&B 本地数据 |

## 训练历史（本次最优模型的形成过程）

| Run | 时间 | 内容 | 最优 |
|---|---|---|---|
| `4_GPU_256x256_20260728_175656` | 07-28 → 07-31 | 从头训练 300 epoch（warmup 5 + flat 150 + cosine 145，LR 3e-5） | ep284，loss ≈ 0.193 |
| `4_GPU_256x256_20260731_074718` | 07-31 07:47 | 续训精修 16 epoch（resume_lr 1e-5） | ep300，loss 0.1899 |
| `4_GPU_256x256_20260731_111654` ⭐ | 07-31 11:16 → 08-01 | **最终精修 100 epoch**（ep301→400，resume_lr 1e-5） | **ep389，loss 0.1878** |

- 最终模型：`best_model.pt` = epoch 389，loss **0.187798**（含 EMA 权重，EMA decay 0.9999）
- 训练终点：epoch 400，loss 0.188581

## 模型与配置

| 项 | 值 |
|---|---|
| 模型 | ConditionalDiT-B（114.9M），img 256×256，patch 8，12 层，hidden 768 |
| 预测目标 | ε-prediction，DDPM，timesteps 1000，cosine schedule |
| 训练 | 4×A40 DDP，batch 48×4=192，AdamW（wd 0.01，clip 1.0） |
| 峰值 LR | 3e-5（首训）/ 1e-5（精修 resume_lr） |
| LR 调度 | warmup 5 + flat + cosine 至 1e-6（三阶段） |
| EMA | decay 0.9999，推理/采样用 EMA 权重 |
| 数据 | data/minst_phase（60,000 张 256×256 相位图） |

## 推理结果

推理链路：随机噪声 → DDPM 1000 步去噪 → 相位全息图 [-π,π] → GS 投影引导（λ_max=0.6，每步对 x₀ 做一次 GS 幅度投影并按 λ_t=λ_max·√ᾱ_t 混合）→ FFT 远场 → 数字图像。

```bash
python inference.py --checkpoint best_run/4_GPU_256x256_20260731_111654/best_model.pt --seed 42
# 默认: --guidance project --guidance-scale 0.6 --img-size 256 --patch-size 8 --digits 0 1 2 3 4 5 6 7 8 9
```

结果图 `recovered_digits_ddpm_1000_389_guided_0.6.png`：0-9 十个数字均清晰可辨，背景散斑经 λ 引导显著抑制。

## 复现训练

```bash
# 若需从零复现该精修过程：
tmux new -s train -d "CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py --config 4_GPU_256x256 > train_$(date +%m%d).log 2>&1"
# 300 epoch 完成后，将 config/4_GPU_256x256.yaml 的 num_epochs 调大，再：
tmux new -s train -d "CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py --config 4_GPU_256x256 --resume checkpoints/4_GPU_256x256_<新时间戳>/checkpoint_latest.pt > train_$(date +%m%d)_ft.log 2>&1"
```
