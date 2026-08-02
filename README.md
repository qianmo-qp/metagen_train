# MNIST Phase Hologram Generation

## 模型简介
这是相位全息数字图生成模型，模型架构采用 Conditional Diffusion Transformer (DiT) ，从 MNIST 官网下载 7万张数字图片。

## 功能说明
扩散模型输入随机噪声，生成 256×256 分辨率的相位全息图，经 FFT 远场衍射后可恢复出对应数字图像。

## 最新最优结果 ⭐

见 [best_run/4_GPU_256x256_20260731_111654/](best_run/4_GPU_256x256_20260731_111654/README.md)：
- **最优模型**：`best_run/4_GPU_256x256_20260731_111654/best_model.pt`（epoch 389，loss 0.1878，EMA 权重）
- **推理结果**：DDPM 1000 步 + GS 投影引导 λ=0.6，0-9 数字清晰可辨
- 含完整训练日志、W&B 数据与复现说明

---

## 模型代码目录结构

```
metagen_train/
├── train.py              # 训练脚本 (单卡 / 多卡 DDP，支持 --resume 续训)
├── inference.py          # 推理脚本 (DDPM 采样 + GS 引导 + FFT 恢复，默认用 EMA 权重)
├── compare_guidance.py   # GS 投影引导 λ 对比工具
├── dit_model.py          # ConditionalDiT 模型架构 (114.9M)
├── diffusion.py          # 扩散过程 (DDPM + DDIM + 引导采样)
├── sample_utils.py       # 采样工具函数（模板构建、GS 投影、增强管线）
├── config.py             # 配置加载器 (读取 YAML)
├── config/               # 训练配置预设
│   ├── 4_GPU_256x256.yaml  # 4×A40 DDP 配置 (256×256)
│   └── 4_GPU_64x64.yaml    # 4×A40 DDP 配置 (64×64)
├── data/
│   └── minst_phase/      # 相位全息图数据 (NPZ 分块)
├── best_run/             # 最优结果归档（模型权重 + 推理图 + 日志 + README）
├── checkpoints/          # 训练产物（按 run 建子目录，不入库）
├── outputs/              # 推理生成结果（按 run 建子目录，不入库）
└── doc/                  # 训练记录文档
```

## 安装python依赖

```bash
pip install torch torchvision numpy matplotlib pillow wandb pyyaml
```

---

## 开始训练

### 4×GPU卡训练，数据分片策略（DDP）

```bash
# 使用 tmux 后台运行 （防止 SSH 断开终端训练）
tmux new -s train -d "CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py --config 4_GPU_256x256 > train_$(date +%m%d).log 2>&1"

# 进入 tmux 查看进度
tmux attach -t train
# 退出 tmux (不中断训练): Ctrl+b 然后按 d
```

产物按 run 归档：`checkpoints/<配置>_<时间戳>/` 与 `outputs/<配置>_<时间戳>/`。

### 端点续训（resume fine-tune）

```bash
# 1) 将 config/4_GPU_256x256.yaml 的 num_epochs 调大（如 300→400）
# 2) 启动（自动加载 model/optimizer/EMA，用 resume_lr=1e-5 对剩余 epoch 重新调度）
tmux new -s train -d "CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py --config 4_GPU_256x256 --resume checkpoints/4_GPU_256x256_<时间戳>/best_model.pt > train_$(date +%m%d)_ft.log 2>&1"
```

## 推理生成图片

```bash
# 默认：DDPM 1000 步 + GS 投影引导 λ=0.6 + EMA 权重，生成数字 0-9
python inference.py --checkpoint best_run/4_GPU_256x256_20260731_111654/best_model.pt --seed 42

# 关闭引导（纯扩散采样）
python inference.py --checkpoint <ckpt> --guidance none

# 其他选项：--guidance-scale 0.4 --img-size 256 --patch-size 8 --digits 0 1 2 3 4 5 6 7 8 9
```

```
随机噪声 → DDPM 1000 步去噪（每步 GS 投影引导 λ_t=λ_max·√ᾱ_t） → 相位全息图 [-π, π]
    → exp(i·phase) → FFT → 振幅 → 归一化 → 数字图像
```
