#!/usr/bin/env python
"""
GPU和PyTorch诊断脚本 (GPU & PyTorch Diagnostic Script)
在GPU服务器上运行这个脚本来检查环境配置
"""

import sys
import torch
import platform

print("=" * 70)
print("PyTorch & GPU 诊断 (Diagnostic)")
print("=" * 70)

# 系统信息
print("\n📊 系统信息 (System Info):")
print(f"  Python: {platform.python_version()}")
print(f"  Platform: {platform.platform()}")

# PyTorch信息
print("\n🔧 PyTorch 信息 (PyTorch Info):")
print(f"  PyTorch Version: {torch.__version__}")
print(f"  Compiled with CUDA: {torch.version.cuda}")
print(f"  Compiled with cuDNN: {torch.backends.cudnn.version()}")

# CUDA可用性
print("\n🎮 CUDA 可用性 (CUDA Availability):")
cuda_available = torch.cuda.is_available()
print(f"  torch.cuda.is_available(): {cuda_available}")

if cuda_available:
    print(f"  ✅ CUDA is available!")
    
    # GPU设备信息
    num_devices = torch.cuda.device_count()
    print(f"\n  GPU设备数量 (Number of GPUs): {num_devices}")
    
    for i in range(num_devices):
        props = torch.cuda.get_device_properties(i)
        print(f"\n  🖥️  GPU {i}:")
        print(f"      Name: {props.name}")
        print(f"      Compute Capability: {props.major}.{props.minor}")
        print(f"      Total Memory: {props.total_memory / 1e9:.2f} GB")
        print(f"      Max Threads per Block: {props.max_threads_per_block}")
        print(f"      Multi-processor Count: {props.multi_processor_count}")
    
    # 当前GPU
    print(f"\n  当前GPU (Current GPU):")
    current_gpu = torch.cuda.current_device()
    print(f"    Device Index: {current_gpu}")
    print(f"    Device Name: {torch.cuda.get_device_name(current_gpu)}")
    
    # 内存信息
    print(f"\n  GPU 内存 (GPU Memory):")
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"    已分配 (Allocated): {allocated:.2f} GB")
    print(f"    已保留 (Reserved): {reserved:.2f} GB")
    print(f"    总容量 (Total): {total:.2f} GB")
    print(f"    可用 (Available): {total - allocated:.2f} GB")
    
    # 测试CUDA操作
    print(f"\n  CUDA 功能测试 (CUDA Function Test):")
    try:
        x = torch.randn(100, 100).cuda()
        y = torch.randn(100, 100).cuda()
        z = torch.matmul(x, y)
        print(f"    ✅ Basic GPU computation works!")
        print(f"    Tensor shape: {z.shape}")
        print(f"    Tensor device: {z.device}")
    except Exception as e:
        print(f"    ❌ GPU computation failed: {e}")
else:
    print("  ❌ CUDA is NOT available!")
    print("\n  解决方案 (Solutions):")
    print("  1. 检查NVIDIA驱动: nvidia-smi")
    print("  2. 重新安装PyTorch with CUDA支持:")
    print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("  3. 确保CUDA工具包已安装")

# 推荐配置
print("\n" + "=" * 70)
print("💡 建议 (Recommendations):")
print("=" * 70)

if cuda_available:
    print("✅ GPU环境已配置正确,可以运行训练")
    print("   运行训练: python train.py")
else:
    print("⚠️  CUDA未检测到")
    print("   选项1: 检查NVIDIA驱动 (check nvidia-smi)")
    print("   选项2: 重新安装PyTorch (reinstall PyTorch with CUDA)")
    print("   选项3: CPU训练 (train on CPU, but slower)")

print("\n" + "=" * 70)
