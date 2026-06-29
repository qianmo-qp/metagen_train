"""
Generate Phase Label Files in IDX Format
从已生成的相位数据目录结构生成MNIST标准格式的标签文件
"""

import numpy as np
import os
import sys
from pathlib import Path
from tqdm import tqdm
import argparse


class PhaseLabelsGenerator:
    """生成相位标签的IDX格式文件"""
    
    def __init__(self, data_dir='./data/minst_phase', verbose=True):
        """
        参数:
            data_dir: 相位数据目录
            verbose: 是否打印进度
        """
        self.data_dir = data_dir
        self.verbose = verbose
    
    def scan_phase_directories(self, split='train'):
        """
        扫描相位数据目录结构,收集标签
        
        参数:
            split: 'train' 或 'test'
        
        返回:
            labels: (N,) 标签数组
            counts: dict，每个数字的计数
        """
        split_dir = os.path.join(self.data_dir, split)
        
        if not os.path.exists(split_dir):
            print(f"❌ 目录不存在: {split_dir}")
            return None, None
        
        all_labels = []
        counts = {i: 0 for i in range(10)}
        
        # 遍历digit_0到digit_9目录
        for digit in range(10):
            digit_dir = os.path.join(split_dir, f'digit_{digit}')
            
            if not os.path.exists(digit_dir):
                if self.verbose:
                    print(f"⚠ 目录不存在: {digit_dir}")
                continue
            
            # 扫描该目录中的.npy文件
            phase_files = sorted([f for f in os.listdir(digit_dir) 
                                 if f.endswith('.npy')])
            
            # 为每个文件添加标签
            all_labels.extend([digit] * len(phase_files))
            counts[digit] = len(phase_files)
        
        labels = np.array(all_labels, dtype=np.uint8)
        return labels, counts
    
    def save_labels_idx(self, labels, output_path):
        """
        保存标签为IDX格式
        
        参数:
            labels: (N,) 标签数组
            output_path: 输出文件路径
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'wb') as f:
            # IDX格式魔数 (0x00000801 for labels)
            magic = np.uint32(0x00000801)
            f.write(magic.astype('>i4').tobytes())
            
            # 标签数量
            num_items = np.uint32(len(labels))
            f.write(num_items.astype('>i4').tobytes())
            
            # 标签数据
            f.write(labels.astype('>u1').tobytes())
        
        if self.verbose:
            print(f"✓ 已保存: {output_path} ({len(labels)} 个标签)")
    
    def save_labels_txt(self, labels, counts, output_path):
        """
        保存标签为CSV格式 (用于调试和验证)
        
        参数:
            labels: (N,) 标签数组
            counts: dict，每个数字的计数
            output_path: 输出文件路径
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            # 写头信息
            f.write("# Phase Labels Index\n")
            f.write(f"# Total: {len(labels)}\n")
            f.write(f"# Format: index,label\n\n")
            
            # 写每个标签
            for idx, label in enumerate(labels):
                f.write(f"{idx},{label}\n")
        
        if self.verbose:
            print(f"✓ 已保存: {output_path}")
    
    def verify_idx_labels(self, idx_path):
        """
        验证IDX标签文件的正确性
        
        参数:
            idx_path: IDX文件路径
        
        返回:
            success: bool
            labels: 读取的标签数组
        """
        try:
            with open(idx_path, 'rb') as f:
                magic = np.frombuffer(f.read(4), dtype='>i4')[0]
                num_items = np.frombuffer(f.read(4), dtype='>i4')[0]
                labels = np.frombuffer(f.read(), dtype='>u1')
            
            if magic != 0x00000801:
                print(f"❌ 魔数错误: {hex(magic)}, 期望: 0x00000801")
                return False, None
            
            if len(labels) != num_items:
                print(f"❌ 标签数量错误: {len(labels)}, 期望: {num_items}")
                return False, None
            
            if self.verbose:
                unique, counts_arr = np.unique(labels, return_counts=True)
                print(f"✓ IDX文件验证通过:")
                print(f"  - 魔数: {hex(magic)}")
                print(f"  - 总数: {num_items}")
                print(f"  - 数字分布: {dict(zip(unique, counts_arr))}")
            
            return True, labels
        
        except Exception as e:
            print(f"❌ 验证失败: {e}")
            return False, None
    
    def generate_labels(self, split='train'):
        """
        为指定的split生成标签文件
        
        参数:
            split: 'train' 或 'test'
        
        返回:
            success: bool
        """
        if self.verbose:
            print(f"\n[{split.upper()}数据集] 开始生成标签文件...")
        
        # 扫描目录
        labels, counts = self.scan_phase_directories(split)
        
        if labels is None:
            print(f"❌ 无法扫描{split}数据集")
            return False
        
        if len(labels) == 0:
            print(f"❌ {split}数据集中未找到任何相位文件")
            return False
        
        # 打印统计
        if self.verbose:
            print(f"\n  标签统计:")
            for digit in range(10):
                if counts[digit] > 0:
                    print(f"    数字{digit}: {counts[digit]:6d} 个")
            print(f"    总计:     {len(labels):6d} 个")
        
        # 保存IDX格式
        split_dir = os.path.join(self.data_dir, split)
        idx_path = os.path.join(split_dir, 'phase_labels.idx1-ubyte')
        self.save_labels_idx(labels, idx_path)
        
        # 保存TXT格式 (用于调试)
        txt_path = os.path.join(split_dir, 'phase_index.txt')
        self.save_labels_txt(labels, counts, txt_path)
        
        # 验证IDX文件
        success, _ = self.verify_idx_labels(idx_path)
        
        return success
    
    def generate_all(self):
        """生成所有split的标签文件"""
        print("=" * 60)
        print("相位标签文件生成 (IDX格式)")
        print("=" * 60)
        
        if self.verbose:
            print(f"\n数据目录: {self.data_dir}")
        
        # 生成训练集标签
        train_ok = self.generate_labels('train')
        
        # 生成测试集标签
        test_ok = self.generate_labels('test')
        
        if train_ok or test_ok:
            print("\n" + "=" * 60)
            print("✓ 标签文件生成完成!")
            print("=" * 60)
            print(f"\n目录结构:")
            print(f"{self.data_dir}/")
            print(f"├── train/")
            print(f"│   ├── digit_0/")
            print(f"│   │   ├── phase_00000.npy")
            print(f"│   │   └── ...")
            print(f"│   ├── phase_labels.idx1-ubyte  ← 新增")
            print(f"│   └── phase_index.txt           ← 新增 (调试用)")
            print(f"└── test/")
            print(f"    ├── digit_0/")
            print(f"    ├── phase_labels.idx1-ubyte  ← 新增")
            print(f"    └── phase_index.txt           ← 新增 (调试用)")
            return True
        else:
            print("\n❌ 标签文件生成失败!")
            return False


def main():
    parser = argparse.ArgumentParser(
        description='为相位数据生成IDX格式标签文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python3 generate_phase_labels.py                     # 使用默认路径
  python3 generate_phase_labels.py -d ./data/phase    # 自定义数据目录
  python3 generate_phase_labels.py -q                 # 安静模式
        ''')
    parser.add_argument('--data-dir', '-d', type=str,
                       default='./data/minst_phase',
                       help='相位数据目录 (默认: ./data/minst_phase)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='安静模式,不显示进度')
    
    args = parser.parse_args()
    
    # 创建生成器
    generator = PhaseLabelsGenerator(
        args.data_dir,
        verbose=not args.quiet
    )
    
    # 生成标签文件
    success = generator.generate_all()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
