"""
Phase Data Cache Builder
将分散的 .npy 文件预处理为高效的 NPZ 缓存文件
"""

import numpy as np
import os
import sys
from pathlib import Path
from tqdm import tqdm
import argparse


class PhaseCacheBuilder:
    """Phase 数据缓存构建器"""
    
    def __init__(self, data_dir='./data/minst_phase', verbose=True):
        """
        参数:
            data_dir: 相位数据根目录
            verbose: 是否打印详细信息
        """
        self.data_dir = data_dir
        self.verbose = verbose
    
    def build_cache(self, split='train', force_rebuild=False, skip_stats=False):
        """
        为指定的 split 构建缓存文件
        
        参数:
            split: 'train' 或 'test'
            force_rebuild: 是否强制重建（即使缓存已存在）
            skip_stats: 是否跳过统计信息计算（加快速度，仅在构建时）
        
        返回:
            cache_path: 缓存文件路径
            success: 是否成功
        """
        split_dir = os.path.join(self.data_dir, split)
        cache_path = os.path.join(split_dir, f'phase_{split}_cache.npz')
        
        # 检查缓存是否已存在
        if os.path.exists(cache_path) and not force_rebuild:
            if self.verbose:
                print(f"✓ 缓存已存在: {cache_path}")
            return cache_path, True
        
        if self.verbose:
            print(f"\n[{split.upper()}] 开始构建缓存...")
        
        # 检查源目录
        if not os.path.exists(split_dir):
            print(f"❌ 目录不存在: {split_dir}")
            return None, False
        
        # 扫描所有 .npy 文件
        all_files = self._scan_files(split_dir)
        
        if not all_files:
            print(f"❌ 未找到任何 .npy 文件在: {split_dir}")
            return None, False
        
        # 加载数据
        phase_data, labels_data = self._load_data(all_files, split, skip_stats=skip_stats)
        
        if phase_data is None:
            return None, False
        
        # 保存缓存
        success = self._save_cache(cache_path, phase_data, labels_data, split)
        
        return cache_path if success else None, success
    
    def _scan_files(self, split_dir):
        """
        扫描目录中的所有 .npy 文件，按数字和文件名排序
        
        返回:
            list: [(digit, file_path), ...] 列表
        """
        all_files = []
        
        for digit in range(10):
            digit_dir = os.path.join(split_dir, f'digit_{digit}')
            if not os.path.exists(digit_dir):
                continue
            
            phase_files = sorted([f for f in os.listdir(digit_dir) 
                                 if f.endswith('.npy')])
            
            for phase_file in phase_files:
                all_files.append((digit, os.path.join(digit_dir, phase_file)))
        
        return all_files
    
    def _load_data(self, all_files, split, skip_stats=False):
        """
        加载所有 .npy 文件到内存
        
        参数:
            all_files: [(digit, file_path), ...] 列表
            split: 'train' 或 'test'
            skip_stats: 是否跳过统计信息计算
        
        返回:
            (phase_data, labels_data) 元组，或 (None, None) 如果失败
        """
        num_samples = len(all_files)
        
        # 预分配数组
        phase_data = np.zeros((num_samples, 256, 256), dtype=np.float16)  # float16 节省 50% 空间
        labels_data = np.zeros(num_samples, dtype=np.int64)
        
        if self.verbose:
            print(f"  加载 {num_samples} 个样本...")
        
        try:
            for idx, (digit, phase_path) in enumerate(
                tqdm(all_files, desc=f"加载{split}数据", unit="file", disable=not self.verbose)
            ):
                phase = np.load(phase_path, allow_pickle=False)
                phase_data[idx] = phase.astype(np.float16)  # 转换为 float16
                labels_data[idx] = digit
            
            if self.verbose:
                print(f"✓ 数据加载完成")
                print(f"  样本数: {num_samples}")
                print(f"  Phase 数组形状: {phase_data.shape}, dtype: {phase_data.dtype}")
                print(f"  Label 数组形状: {labels_data.shape}")
                
                if not skip_stats:
                    # 计算统计信息（可能需要几秒）
                    print(f"  计算统计信息...", end="", flush=True)
                    phase_min = phase_data.min()
                    phase_max = phase_data.max()
                    label_counts = np.bincount(labels_data.astype(int))
                    print(f" 完成")
                    
                    print(f"  Phase 范围: [{phase_min:.4f}, {phase_max:.4f}]")
                    print(f"  Label 分布: {dict(zip(range(10), label_counts))}")
                else:
                    print(f"  (跳过统计计算以加快速度)")
            
            return phase_data, labels_data
        
        except Exception as e:
            print(f"❌ 加载数据失败: {e}")
            return None, None
    
    def _save_cache(self, cache_path, phase_data, labels_data, split):
        """
        保存数据为 NPZ 格式
        
        参数:
            cache_path: 缓存文件路径
            phase_data: Phase 数组
            labels_data: Label 数组
            split: 'train' 或 'test'
        
        返回:
            success: 是否成功
        """
        try:
            if self.verbose:
                print(f"\n  保存缓存...")
            
            # 使用压缩格式节省空间
            np.savez_compressed(
                cache_path,
                phase=phase_data,
                labels=labels_data
            )
            
            # 计算文件大小
            cache_size_mb = os.path.getsize(cache_path) / (1024 * 1024)
            
            if self.verbose:
                print(f"✓ 缓存保存成功: {cache_path}")
                print(f"  文件大小: {cache_size_mb:.2f} MB")
            
            return True
        
        except Exception as e:
            print(f"❌ 保存缓存失败: {e}")
            return False
    
    def verify_cache(self, split='train'):
        """
        验证缓存文件的完整性和正确性
        
        参数:
            split: 'train' 或 'test'
        
        返回:
            success: 是否验证通过
        """
        split_dir = os.path.join(self.data_dir, split)
        cache_path = os.path.join(split_dir, f'phase_{split}_cache.npz')
        
        if not os.path.exists(cache_path):
            print(f"❌ 缓存文件不存在: {cache_path}")
            return False
        
        try:
            if self.verbose:
                print(f"\n[验证] {split} 缓存...")
            
            # 加载缓存
            cache = np.load(cache_path)
            phase = cache['phase']
            labels = cache['labels']
            
            # 检查形状和类型
            assert phase.ndim == 3, f"Phase 维度错误: {phase.ndim}, 期望: 3"
            assert phase.shape[1:] == (256, 256), f"Phase 分辨率错误: {phase.shape[1:]}"
            assert phase.dtype == np.float16, f"Phase dtype 错误: {phase.dtype}, 期望: float16"
            
            assert labels.ndim == 1, f"Labels 维度错误: {labels.ndim}"
            assert len(labels) == len(phase), f"样本数不匹配"
            assert labels.dtype == np.int64, f"Labels dtype 错误: {labels.dtype}"
            
            # 检查标签范围
            assert labels.min() >= 0 and labels.max() <= 9, f"Label 值超出范围: [{labels.min()}, {labels.max()}]"
            
            if self.verbose:
                print(f"✓ 缓存验证通过:")
                print(f"  样本数: {len(phase)}")
                print(f"  Phase 形状: {phase.shape}, dtype: {phase.dtype}")
                print(f"  Labels 形状: {labels.shape}, dtype: {labels.dtype}")
                print(f"  Phase 范围: [{phase.min():.4f}, {phase.max():.4f}]")
                print(f"  Label 分布: {np.bincount(labels)}")
            
            return True
        
        except Exception as e:
            print(f"❌ 缓存验证失败: {e}")
            return False
    
    def build_all(self, force_rebuild=False, skip_stats=True):
        """
        为所有 split 构建缓存
        
        参数:
            force_rebuild: 是否强制重建
            skip_stats: 是否跳过统计计算 (〜60K样本更快)
        
        返回:
            success: 是否全部成功
        """
        print("=" * 60)
        print("Phase 数据缓存构建")
        print("=" * 60)
        print(f"\n数据目录: {self.data_dir}")
        
        results = {}
        
        for split in ['train', 'test']:
            cache_path, success = self.build_cache(split, force_rebuild, skip_stats=skip_stats)
            results[split] = (cache_path, success)
            
            if success:
                self.verify_cache(split)
        
        # 汇总
        print("\n" + "=" * 60)
        print("构建完成!")
        print("=" * 60)
        
        for split, (cache_path, success) in results.items():
            status = "✓" if success else "❌"
            print(f"{status} {split.upper()}: {cache_path}")
        
        return all(success for _, success in results.values())


def main():
    parser = argparse.ArgumentParser(
        description='Phase 数据缓存构建工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python3 phase_cache_builder.py                       # 使用默认路径
  python3 phase_cache_builder.py -d /path/to/data     # 自定义数据目录
  python3 phase_cache_builder.py -f                   # 强制重建所有缓存
  python3 phase_cache_builder.py -v train             # 只构建 train 缓存
        ''')
    
    parser.add_argument('--data-dir', '-d', type=str,
                       default='./data/minst_phase',
                       help='Phase 数据目录 (默认: ./data/minst_phase)')
    parser.add_argument('--split', '-s', type=str, default=None,
                       choices=['train', 'test'],
                       help='指定构建的 split (默认: 全部)')
    parser.add_argument('--force', '-f', action='store_true',
                       help='强制重建缓存 (即使已存在)')
    parser.add_argument('--verify-only', '-v', action='store_true',
                       help='仅验证现有缓存，不构建')
    parser.add_argument('--skip-stats', action='store_true',
                       help='跳过统计计算 (加快速构建, 默认含体方案)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='安静模式')
    
    args = parser.parse_args()
    
    # 创建构建器
    builder = PhaseCacheBuilder(args.data_dir, verbose=not args.quiet)
    
    # 执行操作
    if args.verify_only:
        if args.split:
            success = builder.verify_cache(args.split)
        else:
            success = all(builder.verify_cache(split) for split in ['train', 'test'])
    else:
        if args.split:
            _, success = builder.build_cache(args.split, args.force, skip_stats=args.skip_stats)
        else:
            success = builder.build_all(args.force, skip_stats=args.skip_stats)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
