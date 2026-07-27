# MNIST Phase Hologram Generation

## 模型简介：
这是相位全息数字图生成模型，模型架构采用 Conditional Diffusion Transformer (DiT) ，从 MNIST 官网下载 7万张数字图片。

## 功能说明：
扩散模型输入随机噪声，生成 256×256 分辨率的相位全息图，经 FFT 远场衍射后可恢复出对应数字图像。

---

## 模型代码目录结构

```
metagen_train/
├── train.py              # 训练脚本 (支持单卡 / 多卡 DDP)
├── inference.py          # 推理脚本 (DDPM 采样 + FFT 恢复)
├── dit_model.py          # ConditionalDiT 模型架构 (114.9M)
├── diffusion.py          # 扩散过程 (DDPM + DDIM)
├── sample_utils.py       # 采样工具函数
├── config.py             # 配置加载器 (读取 YAML)
├── config/               # 训练配置预设
│   ├── 4_GPU_256x256.yaml  # 4×A40 DDP数据分片配置 (256×256)
│   └── 1_GPU_256x256.yaml  # 1×A40 单卡配置 (256×256)
├── data/
│   └── minst_phase/      # 相位全息图数据 (NPZ 分块)
│       ├── train/        # 60,000 训练样本
│       └── test/         # 10,000 测试样本
├── checkpoints/          # 模型权重
│   ├── checkpoint_latest.pt
│   └── best_model.pt
├── outputs/              # 推理生成结果
├── doc/                  # 训练记录文档
└── wandb/                # W&B 本地缓存
```

---
## 安装python依赖

```bash
pip install torch torchvision numpy matplotlib pillow wandb pyyaml
```

---

## 开始训练

### 单张GPU卡训练

```bash
# 使用后台运行训练程序（防止 SSH 断开终端训练）
nohup python -u train.py --config 1_GPU_256x256 > train.log 2>&1 &
```

### 4×GPU卡训练，数据分片策略（DDP）

```bash
# 使用 tmux 后台运行 （防止 SSH 断开终端训练）
tmux new -s train -d "CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py --config 4_GPU_256x256 > train.log 2>&1"

# 进入 tmux 查看进度
tmux attach -t train

# 退出 tmux (不中断训练): Ctrl+b 然后按 d
```

### 端点续训，即从某个 Checkpoint 继续训练

```bash
# 加载模型权重，重新初始化 optimizer/scheduler
torchrun --nproc_per_node=4 train.py --config 4_GPU_256x256 \
  --resume checkpoints/best/checkpoint_latest.pt
```


## 推理生成图片

```bash

# 指定数字 1-9 的图像，FFT恢复的数字图片保存到文件夹 outputs/
# 生成 64×64 分辨率图像，保留更多细节
python inference.py --checkpoint checkpoints/best/best_0718_4GPU.pt \
  --resolution 64 --contrast 1.5 --gamma 0.7 --sharpen 1.0
```

```
随机噪声 → DDPM 1000 步去噪 → 相位全息图 [-π, π]
    → exp(i·phase) → FFT → 振幅 → 归一化 → 数字图像
```
