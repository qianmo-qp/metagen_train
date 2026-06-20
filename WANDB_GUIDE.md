# W&B 监控指南 (W&B Monitoring Guide)

## 概述

已为训练脚本集成了 W&B (Weights & Biases) 监控功能,可以实时跟踪:
- ✅ 训练损失曲线
- ✅ 每10个epoch生成的样本图片
- ✅ Epoch进度和指标
- ✅ 训练配置

## 🚀 快速开始

### 步骤1: 拉取最新代码

```bash
cd /mnt/model_data/qp/metagen_train
git pull origin minst_test
```

### 步骤2: 启动训练

```bash
# 后台运行训练 (带W&B监控)
nohup python -u train.py > train.log 2>&1 &

# 记下进程ID
# [1] 12345
```

**预期输出:**
```
✅ CUDA is available
Device: Tesla V100
GPU Memory: 16.00 GB

📊 Initializing W&B...
✅ W&B initialized

Loading MNIST dataset...
Creating model...
Training on cuda
Total epochs: 100
```

### 步骤3: 访问W&B仪表板

打开浏览器访问:
```
https://wandb.ai/[your-username]/minst
```

你会看到实时监控的:
- 训练损失图表
- 样本图片库
- 超参数配置

## 📊 监控指标

### Loss (训练损失)
- **实时更新**: 每100步更新一次
- **期望趋势**: 从0.2-0.3逐步下降到0.001-0.01
- **何时良好**: 损失稳定下降表示学习正常

### Samples (生成样本)
- **更新频率**: 每10个epoch生成一次
- **显示内容**: 10个数字类别各1个样本
- **何时良好**: 样本从噪声逐步变清晰,类别特征明显

### Epoch Loss (Epoch平均损失)
- **显示**: 每个epoch的平均损失
- **用途**: 查看epoch之间的长期趋势

## 📈 训练配置

```
num_epochs: 100       # 100个epoch (原来10个)
batch_size: 128       # 批大小
learning_rate: 1e-4   # 学习率
timesteps: 1000       # 扩散步数
sample_interval: 10   # 每10个epoch采样
```

## 🎯 关键检查点

### Epoch 10
- 损失应该降到: ~0.10-0.15
- 样本应该: 开始有可识别的轮廓

### Epoch 30
- 损失应该降到: ~0.03-0.05
- 样本应该: 清晰,类别区分明显

### Epoch 50
- 损失应该降到: ~0.01-0.02
- 样本应该: 高质量,很少噪声

### Epoch 100
- 损失应该降到: ~0.001-0.005
- 样本应该: 最优质量

## 🔍 在W&B上查看数据

### 1. 查看损失曲线

```
Charts → loss
- 显示实时训练损失
- X轴: step (总步数)
- Y轴: loss值
```

### 2. 查看生成样本

```
Media → samples_epoch_*
- 显示每10个epoch的生成图片
- 10行,每行1个样本
- 数字从0到9
```

### 3. 查看配置

```
Overview → Config
- 显示所有超参数
- 显示模型架构信息
```

## 💡 常见操作

### 下载样本图片

```bash
# 从W&B下载样本
wandb sync --clean

# 或手动下载 (在W&B网页上)
# 1. 点击样本图片
# 2. 点击下载按钮
```

### 对比不同运行

```bash
# W&B支持多个run对比
1. 创建一个Team Project
2. 运行多个训练
3. W&B自动显示对比图表
```

### 导出数据

```bash
# 导出为CSV
1. 在W&B网页上
2. 点击图表右上角的"..."
3. 选择"Download as CSV"
```

## 📝 日志文件

训练产生的日志:

```
train.log              # 完整训练日志
outputs/
├── generated_samples_epoch_10.png
├── generated_samples_epoch_20.png
├── ...
└── generated_samples_epoch_100.png
checkpoints/
├── checkpoint_epoch_1.pt
├── checkpoint_epoch_2.pt
├── ...
└── checkpoint_epoch_100.pt
```

## 🚨 故障排除

### Q: W&B连接失败

```bash
# 检查网络
ping wandb.ai

# 重新初始化
wandb login --relogin

# 或设置API key
export WANDB_API_KEY="wandb_v1_..."
```

### Q: 看不到样本图片

```bash
# 可能原因:
# 1. 采样还未进行 (< epoch 10)
# 2. GPU内存不足导致采样失败

# 检查日志
tail -100 train.log | grep "Generating"
```

### Q: 训练太慢

```bash
# 检查GPU使用
nvidia-smi

# 如果GPU使用率低:
# 1. 增加batch_size (128 → 256)
# 2. 减少采样频率 (10 → 20)
# 3. 使用混合精度训练
```

## 📚 参考

- **W&B官网**: https://wandb.ai
- **W&B文档**: https://docs.wandb.ai
- **Project Link**: https://wandb.ai/[your-username]/minst

## 🎉 预期结果

100个epoch训练后,你应该看到:

```
✅ Loss: 0.001-0.005 (优秀!)
✅ Samples: 清晰,可识别的数字
✅ All Classes: 0-9都有良好表现
✅ W&B Dashboard: 完整的训练曲线记录
```

---

**开始训练吧!** 🚀

```bash
nohup python -u train.py > train.log 2>&1 &
```

然后访问: https://wandb.ai/[your-username]/minst
