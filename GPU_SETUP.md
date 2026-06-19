# GPU 设置和故障排除指南

## 问题诊断

### 问题1: ImportError (ONNX)

**错误信息:**
```
ImportError: cannot import name 'ExportOptions' from 'torch.onnx._internal.exporter'
```

**原因:** PyTorch版本兼容性问题

**解决方案:**

已在 `train.py` 中添加了修复:
```python
os.environ['TORCH_DISABLE_DYNANMO'] = '1'
```

现在运行:
```bash
python train.py
```

### 问题2: 没有检测到GPU (Using device: cpu)

**原因:** 
1. NVIDIA驱动未安装或版本不对
2. CUDA未安装
3. PyTorch未用CUDA支持编译

**检查步骤:**

#### 步骤1: 检查硬件
```bash
# 查看GPU硬件
nvidia-smi

# 输出应该显示:
# - GPU型号 (e.g., Tesla V100)
# - GPU驱动版本
# - GPU使用情况
```

#### 步骤2: 运行诊断脚本
```bash
# 新增的诊断脚本
python check_gpu.py

# 输出应该包括:
# - PyTorch版本
# - CUDA编译版本
# - GPU设备列表
# - 内存信息
# - CUDA功能测试
```

#### 步骤3: 检查PyTorch CUDA支持
```bash
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.version.cuda)"
python -c "import torch; print(torch.cuda.device_count())"
```

### 解决方案

#### 方案A: 重新安装PyTorch with CUDA支持

```bash
# 卸载当前PyTorch
pip uninstall torch torchvision torchaudio -y

# 安装CUDA 11.8版本的PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 或CUDA 12.1版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 方案B: 安装NVIDIA驱动

```bash
# 检查当前驱动
nvidia-smi

# 如果没有, 需要安装 (具体命令取决于系统)
# Ubuntu/Debian:
sudo apt-get install nvidia-driver-XXX

# CentOS/RHEL:
sudo yum install nvidia-driver-XXX
```

#### 方案C: 使用CPU训练 (如果GPU不可用)

```bash
# 代码会自动检测并使用CPU
# 只是速度会慢很多
python train.py

# 预计速度:
# - GPU (V100): ~30秒/epoch
# - GPU (A100): ~10秒/epoch
# - CPU: ~2-3分钟/epoch
```

## 完整训练流程 (在GPU服务器上)

### 1. 诊断环境
```bash
python check_gpu.py
```

预期输出:
```
✅ CUDA is available!
🖥️  GPU 0:
    Name: Tesla V100
    Total Memory: 16.00 GB
    ✅ Basic GPU computation works!
```

### 2. 启动训练 (后台运行)

```bash
# 方式1: 使用nohup (推荐)
nohup python -u train.py > train.log 2>&1 &

# 方式2: 使用screen或tmux
screen -S training
python train.py
# 按Ctrl+A+D离开screen
```

### 3. 监控训练

```bash
# 查看日志
tail -f train.log

# 或查看最后100行
tail -100 train.log

# 查看GPU使用情况
watch -n 1 nvidia-smi
```

### 4. 训练参数 (在train.py中调整)

```python
# 第163行左右
num_epochs = 10          # 增加到20-50获得更好质量
batch_size = 128         # 减小到64/32以节省内存
learning_rate = 1e-4
timesteps = 1000
```

### 5. GPU内存优化

如果GPU内存不足:

```python
# 减少batch_size
batch_size = 64  # or 32

# 或减少模型大小
model = ConditionalDiT(
    hidden_dim=128,      # 从192减到128
    num_layers=4,        # 从6减到4
    num_heads=2,         # 从3减到2
)
```

## 预期性能

### 训练速度
| 设备 | 每个Epoch时间 | 10 Epochs总时间 |
|------|---|---|
| NVIDIA V100 | ~30秒 | ~5分钟 |
| NVIDIA A100 | ~10秒 | ~2分钟 |
| CPU (8核) | ~2-3分钟 | ~20-30分钟 |
| M1/M2 Mac | ~1-2分钟 | ~10-20分钟 |

### 采样速度
| 设备 | 1000步采样时间 |
|------|---|
| GPU | ~5分钟 |
| CPU | ~30分钟 |

## 常见问题

**Q: 运行训练后看不到GPU被使用**
```bash
# 检查
nvidia-smi

# 应该看到 "python" 进程占用GPU内存
```

**Q: CUDA out of memory**
```bash
# 减少batch_size
batch_size = 32

# 或清理GPU内存
python -c "import torch; torch.cuda.empty_cache()"
```

**Q: 想在CPU和GPU之间测试**
```bash
# 在train.py中手动设置设备
device = 'cuda'  # 或 'cpu'

# 然后运行
python train.py
```

**Q: 如何检查GPU是否正常工作**
```bash
# 运行诊断脚本
python check_gpu.py

# 应该输出 ✅ CUDA is available!
```

## 文件清单

- `train.py` - 修复了torch._dynamo问题,改进了GPU检测
- `check_gpu.py` - 新增GPU诊断脚本
- `GPU_SETUP.md` - 本文件

## 下一步

1. 运行 `python check_gpu.py` 诊断环境
2. 根据输出结果采取相应措施
3. 运行 `python train.py` 开始训练
4. 使用 `python sample.py` 生成样本

---

有问题? 检查日志文件 `train.log` 获取详细信息。
