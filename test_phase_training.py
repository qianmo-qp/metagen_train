"""
Quick test for phase hologram training
验证相位数据能否正确加载并用于训练
"""

import torch
import numpy as np
import os
from pathlib import Path

# 检查相位数据
def check_phase_data(data_dir='data/minst_phase'):
    """检查相位数据是否存在且格式正确"""
    print("="*60)
    print("检查相位数据")
    print("="*60)
    
    if not os.path.exists(data_dir):
        print(f"❌ 目录不存在: {data_dir}")
        return False
    
    # 统计文件
    total_files = 0
    for split in ['train', 'test']:
        split_dir = os.path.join(data_dir, split)
        if os.path.exists(split_dir):
            split_count = 0
            for digit in range(10):
                digit_dir = os.path.join(split_dir, f'digit_{digit}')
                if os.path.exists(digit_dir):
                    files = [f for f in os.listdir(digit_dir) if f.endswith('.npy')]
                    split_count += len(files)
                    if files:
                        print(f"  {split}/digit_{digit}: {len(files)} 个文件")
            
            print(f"  {split.upper()} 总计: {split_count} 个文件")
            total_files += split_count
    
    print(f"\n✓ 总共: {total_files} 个相位文件")
    
    # 检查文件格式
    print("\n检查文件格式...")
    sample_file = os.path.join(data_dir, 'train/digit_0/phase_00000.npy')
    if os.path.exists(sample_file):
        phase = np.load(sample_file)
        print(f"  形状: {phase.shape}")
        print(f"  数据类型: {phase.dtype}")
        print(f"  范围: [{phase.min():.4f}, {phase.max():.4f}]")
        print(f"  ✓ 文件格式正确")
        return True
    else:
        print(f"❌ 样本文件不存在: {sample_file}")
        return False


# 测试数据加载
def test_data_loading():
    """测试新的数据加载函数"""
    print("\n" + "="*60)
    print("测试数据加载")
    print("="*60)
    
    # 导入训练脚本中的函数
    try:
        import sys
        sys.path.insert(0, '/Users/qp/workspace/metagen_train')
        from train import load_phase_data, normalize_phase_data
        
        print("加载相位数据...")
        train_phase, train_labels, test_phase, test_labels = load_phase_data()
        
        print(f"✓ 训练集加载成功:")
        print(f"  形状: {train_phase.shape}")
        print(f"  标签: {train_labels.shape}")
        
        print(f"\n正规化相位数据...")
        train_phase_norm = normalize_phase_data(train_phase)
        
        print(f"✓ 正规化成功:")
        print(f"  形状: {train_phase_norm.shape}")
        print(f"  范围: [{train_phase_norm.min():.4f}, {train_phase_norm.max():.4f}]")
        print(f"  数据类型: {train_phase_norm.dtype}")
        
        return True
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return False


# 测试模型输入
def test_model_input():
    """测试模型是否能接受256×256的相位数据"""
    print("\n" + "="*60)
    print("测试模型输入")
    print("="*60)
    
    try:
        from dit_model import ConditionalDiT
        
        print("创建模型 (256×256, patch_size=64)...")
        model = ConditionalDiT(
            img_size=256,
            patch_size=64,
            in_channels=1,
            hidden_dim=192,
            num_heads=3,
            num_layers=6,
            time_dim=256,
            num_classes=10,
            mlp_ratio=4,
        )
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"✓ 模型创建成功: {num_params:,} 参数")
        
        # 测试前向传播
        print("\n测试前向传播...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        model.eval()
        
        # 创建虚拟输入
        batch_size = 2
        x = torch.randn(batch_size, 1, 256, 256, device=device)
        t = torch.tensor([100, 200], device=device)
        c = torch.tensor([0, 5], device=device)
        
        with torch.no_grad():
            output = model(x, t, c)
        
        print(f"✓ 前向传播成功:")
        print(f"  输入: {x.shape}")
        print(f"  输出: {output.shape}")
        print(f"  预期输出形状: {x.shape}")
        
        if output.shape == x.shape:
            print(f"  ✓ 形状匹配!")
            return True
        else:
            print(f"  ❌ 形状不匹配!")
            return False
            
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "█"*60)
    print("相位全息图训练测试")
    print("█"*60)
    
    results = []
    
    # 检查数据
    data_ok = check_phase_data()
    results.append(("相位数据检查", data_ok))
    
    if data_ok:
        # 测试加载
        load_ok = test_data_loading()
        results.append(("数据加载测试", load_ok))
    
    # 测试模型
    model_ok = test_model_input()
    results.append(("模型输入测试", model_ok))
    
    # 总结
    print("\n" + "█"*60)
    print("测试总结")
    print("█"*60)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}  {test_name}")
    
    all_pass = all(result for _, result in results)
    if all_pass:
        print("\n✨ 所有测试通过! 可以开始训练!")
        print("\n开始训练:")
        print("  python train.py")
        print("\n后台训练:")
        print("  nohup python3 train.py > train.log 2>&1 &")
    else:
        print("\n⚠️ 有测试失败,请检查数据和环境")
    
    return all_pass


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
