"""
MNIST Image to Phase Hologram Processing Pipeline
MNIST 图像 → 相位全息图转换 → NPZ 缓存生成（一体化处理）

整合功能：
1. MNIST IDX 格式读取
2. 图像 → 相位全息图转换 (Gerchberg-Saxton)
3. 相位数据存储为单个 .npy 文件
4. 自动生成 NPZ 缓存（float16 压缩存储）
"""

import numpy as np
import os
import sys
from pathlib import Path
from tqdm import tqdm
import argparse
import math

# 导入相位提取器
from phase_extractor import PhaseExtractor


class MNISTPhaseConverter:
    """MNIST 图像到相位全息图的完整转换管道"""
    
    def __init__(self, mnist_data_dir, output_dir, resolution=256, iterations=200, verbose=True):
        """
        参数:
            mnist_data_dir: MNIST 数据集目录 (包含 train-images.idx3-ubyte 等)
            output_dir: 输出目录
            resolution: 相位处理分辨率 (默认: 256)
            iterations: Gerchberg-Saxton 迭代次数 (默认: 200)
            verbose: 是否打印详细信息
        """
        self.mnist_data_dir = mnist_data_dir
        self.output_dir = output_dir
        self.resolution = resolution
        self.iterations = iterations
        self.verbose = verbose
        self.extractor = PhaseExtractor(resolution, iterations, verbose=False)
    
    # ======================== IDX 格式读取 ========================
    
    def read_idx_images(self, filename):
        """
        读取 MNIST IDX 格式图像文件
        
        IDX 格式: magic_number (4B) | num_images (4B) | height (4B) | width (4B) | data
        
        参数:
            filename: IDX 文件路径
        
        返回:
            images: (N, 28, 28) 数组
        """
        with open(filename, 'rb') as f:
            magic = np.frombuffer(f.read(4), dtype='>i4')[0]
            num_images = np.frombuffer(f.read(4), dtype='>i4')[0]
            height = np.frombuffer(f.read(4), dtype='>i4')[0]
            width = np.frombuffer(f.read(4), dtype='>i4')[0]
            
            if self.verbose:
                print(f"  IDX 魔数: 0x{magic:08x}")
                print(f"  图像数量: {num_images}")
                print(f"  图像大小: {height}×{width}")
            
            images = np.frombuffer(f.read(), dtype='>u1').reshape(num_images, height, width)
            return images
    
    def read_idx_labels(self, filename):
        """
        读取 MNIST IDX 格式标签文件
        
        IDX 格式: magic_number (4B) | num_items (4B) | data
        
        参数:
            filename: IDX 文件路径
        
        返回:
            labels: (N,) 数组，值为 0-9
        """
        with open(filename, 'rb') as f:
            magic = np.frombuffer(f.read(4), dtype='>i4')[0]
            num_items = np.frombuffer(f.read(4), dtype='>i4')[0]
            
            labels = np.frombuffer(f.read(), dtype='>u1')
            return labels
    
    # ======================== 图像到相位转换 ========================
    
    def image_to_phase(self, image_28x28):
        """
        将单个 MNIST 图像转换为相位全息图
        
        流程:
        1. 归一化到 [0, 1]
        2. 上采样到 256×256
        3. 对比度增强
        4. Gerchberg-Saxton 算法提取相位
        
        参数:
            image_28x28: (28, 28) 灰度图像，像素值 [0, 255]
        
        返回:
            phase: (256, 256) 相位全息图，范围 [-π, π]
        """
        from PIL import Image as PILImage
        
        # 1. 归一化到 [0, 1]
        img_array = image_28x28.astype(float) / 255.0
        
        # 2. 上采样到处理分辨率
        img_pil = PILImage.fromarray((img_array * 255).astype(np.uint8))
        img_pil = img_pil.resize((self.resolution, self.resolution), PILImage.Resampling.LANCZOS)
        img_array = np.array(img_pil).astype(float) / 255.0
        
        # 3. 对比度增强
        img_array = (img_array - 0.5) * 1.3 + 0.5
        img_array = np.clip(img_array, 0, 1)
        
        # 4. GS 算法提取相位
        phase, _ = self.extractor.gs_algorithm(img_array)
        
        return phase
    
    # ======================== 数据集处理 ========================
    
    def process_dataset(self, split='train'):
        """
        处理单个数据集 (train 或 test)
        
        步骤:
        1. 加载 MNIST IDX 文件
        2. 逐个转换为相位
        3. 按数字类别保存为 .npy 文件
        
        参数:
            split: 'train' 或 'test'
        
        返回:
            success: 是否处理成功
        """
        # 确定文件路径
        if split == 'train':
            images_file = os.path.join(self.mnist_data_dir, 'train-images.idx3-ubyte')
            labels_file = os.path.join(self.mnist_data_dir, 'train-labels.idx1-ubyte')
        elif split == 'test':
            images_file = os.path.join(self.mnist_data_dir, 't10k-images.idx3-ubyte')
            labels_file = os.path.join(self.mnist_data_dir, 't10k-labels.idx1-ubyte')
        else:
            raise ValueError(f"split 必须是 'train' 或 'test'，得到 {split}")
        
        # 检查文件
        if not os.path.exists(images_file):
            print(f"❌ 文件不存在: {images_file}")
            return False
        if not os.path.exists(labels_file):
            print(f"❌ 文件不存在: {labels_file}")
            return False
        
        if self.verbose:
            print(f"\n[{split.upper()}数据集] 开始处理...")
        
        # 读取 MNIST IDX 格式
        images = self.read_idx_images(images_file)
        labels = self.read_idx_labels(labels_file)
        
        if self.verbose:
            print(f"  加载 {len(images)} 个图像")
            print(f"  标签范围: [{labels.min()}, {labels.max()}]")
        
        # 创建输出目录
        output_split_dir = os.path.join(self.output_dir, split)
        os.makedirs(output_split_dir, exist_ok=True)
        
        # 创建标签目录 (0-9)
        for digit in range(10):
            os.makedirs(os.path.join(output_split_dir, f'digit_{digit}'), exist_ok=True)
        
        # 处理每个图像
        processed_count = {i: 0 for i in range(10)}
        
        for idx in tqdm(range(len(images)), desc=f"处理{split}数据集", disable=not self.verbose):
            image = images[idx]
            label = labels[idx]
            
            # 转换为相位
            phase = self.image_to_phase(image)
            
            # 保存相位为 .npy 文件
            digit_dir = os.path.join(output_split_dir, f'digit_{label}')
            phase_filename = f'phase_{processed_count[label]:05d}.npy'
            phase_path = os.path.join(digit_dir, phase_filename)
            
            np.save(phase_path, phase)
            
            processed_count[label] += 1
        
        # 统计信息
        if self.verbose:
            print(f"\n✓ {split.upper()} 处理完成!")
            total = sum(processed_count.values())
            print(f"  总数: {total}")
            print(f"  各数字分布:")
            for digit in range(10):
                print(f"    数字 {digit}: {processed_count[digit]:6d} 个")
        
        return True
    
    # ======================== NPZ 缓存生成 ========================
    
    def build_cache(self, split='train', skip_stats=True):
        """
        为指定的 split 构建 NPZ 缓存文件
        
        特性:
        - float16 存储（节省 50% 空间）
        - 使用 savez_compressed
        - 验证完整性
        
        参数:
            split: 'train' 或 'test'
            skip_stats: 是否跳过统计计算（加快速度）
        
        返回:
            cache_path: 缓存文件路径
            success: 是否成功
        """
        split_dir = os.path.join(self.output_dir, split)
        cache_path = os.path.join(split_dir, f'phase_{split}_cache.npz')
        
        if self.verbose:
            print(f"\n[{split.upper()}] 开始构建 NPZ 缓存...")
        
        # 检查源目录
        if not os.path.exists(split_dir):
            print(f"❌ 目录不存在: {split_dir}")
            return None, False
        
        # 扫描所有 .npy 文件
        all_files = []
        for digit in range(10):
            digit_dir = os.path.join(split_dir, f'digit_{digit}')
            if os.path.exists(digit_dir):
                phase_files = sorted([f for f in os.listdir(digit_dir) 
                                     if f.endswith('.npy')])
                for phase_file in phase_files:
                    all_files.append((digit, os.path.join(digit_dir, phase_file)))
        
        if not all_files:
            print(f"❌ 未找到任何 .npy 文件在: {split_dir}")
            return None, False
        
        # 加载所有数据到内存
        num_samples = len(all_files)
        phase_data = np.zeros((num_samples, 256, 256), dtype=np.float16)  # float16 节省空间
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
                print(f"  Label 数组形状: {labels_data.shape}, dtype: {labels_data.dtype}")
                
                if not skip_stats:
                    print(f"  计算统计信息...", end="", flush=True)
                    phase_min = phase_data.min()
                    phase_max = phase_data.max()
                    label_counts = np.bincount(labels_data.astype(int))
                    print(f" 完成")
                    print(f"  Phase 范围: [{phase_min:.4f}, {phase_max:.4f}]")
                    print(f"  Label 分布: {dict(zip(range(10), label_counts))}")
            
        except Exception as e:
            print(f"❌ 加载数据失败: {e}")
            return None, False
        
        # 保存 NPZ 缓存
        try:
            if self.verbose:
                print(f"  保存 NPZ 缓存...", end="", flush=True)
            
            np.savez_compressed(cache_path, phase=phase_data, labels=labels_data)
            
            cache_size_mb = os.path.getsize(cache_path) / (1024 * 1024)
            
            if self.verbose:
                print(f" 完成")
                print(f"✓ 缓存保存成功: {cache_path}")
                print(f"  文件大小: {cache_size_mb:.2f} MB")
            
            return cache_path, True
        
        except Exception as e:
            print(f"❌ 保存缓存失败: {e}")
            return None, False
    
    def verify_cache(self, split='train'):
        """
        验证缓存文件的完整性和正确性
        
        参数:
            split: 'train' 或 'test'
        
        返回:
            success: 是否验证通过
        """
        split_dir = os.path.join(self.output_dir, split)
        cache_path = os.path.join(split_dir, f'phase_{split}_cache.npz')
        
        if not os.path.exists(cache_path):
            print(f"❌ 缓存文件不存在: {cache_path}")
            return False
        
        try:
            if self.verbose:
                print(f"\n[验证] {split} 缓存...")
            
            cache = np.load(cache_path)
            phase = cache['phase']
            labels = cache['labels']
            
            # 检查形状和类型
            assert phase.ndim == 3, f"Phase 维度错误: {phase.ndim}"
            assert phase.shape[1:] == (256, 256), f"Phase 分辨率错误: {phase.shape[1:]}"
            assert phase.dtype == np.float16, f"Phase dtype 错误: {phase.dtype}"
            
            assert labels.ndim == 1, f"Labels 维度错误: {labels.ndim}"
            assert len(labels) == len(phase), f"样本数不匹配"
            assert labels.dtype == np.int64, f"Labels dtype 错误: {labels.dtype}"
            
            assert labels.min() >= 0 and labels.max() <= 9, f"Label 超出范围: [{labels.min()}, {labels.max()}]"
            
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
    
    # ======================== 完整流程 ========================
    
    def process_all(self, build_cache=True):
        """
        完整的处理流程:
        1. MNIST IDX → 相位 .npy 文件
        2. 生成 NPZ 缓存 (可选)
        
        参数:
            build_cache: 是否在完成后构建 NPZ 缓存
        
        返回:
            success: 是否全部成功
        """
        print("=" * 60)
        print("MNIST 图像 → 相位全息图转换管道")
        print("=" * 60)
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        if self.verbose:
            print(f"\n配置:")
            print(f"  MNIST 目录: {self.mnist_data_dir}")
            print(f"  输出目录: {self.output_dir}")
            print(f"  分辨率: {self.resolution}×{self.resolution}")
            print(f"  GS 迭代次数: {self.iterations}")
            print(f"  构建缓存: {build_cache}")
        
        # 处理训练集和测试集
        train_ok = self.process_dataset('train')
        test_ok = self.process_dataset('test')
        
        if not (train_ok and test_ok):
            print("\n❌ 数据集处理失败!")
            return False
        
        print("\n" + "=" * 60)
        print("✓ 所有数据集处理完成!")
        print("=" * 60)
        
        # 构建缓存
        if build_cache:
            print("\n" + "=" * 60)
            print("开始构建 NPZ 缓存...")
            print("=" * 60)
            
            results = {}
            for split in ['train', 'test']:
                cache_path, success = self.build_cache(split, skip_stats=True)
                results[split] = (cache_path, success)
                
                if success:
                    self.verify_cache(split)
            
            if not all(success for _, success in results.values()):
                print("\n⚠ 缓存构建失败!")
                return False
        
        print("\n" + "=" * 60)
        print("✓ 完整处理流程完成!")
        print("=" * 60)
        print(f"\n目录结构:")
        print(f"{self.output_dir}/")
        print(f"├── train/")
        print(f"│   ├── digit_0/ ... digit_9/  (共 60,000 个 .npy 文件)")
        print(f"│   └── phase_train_cache.npz  (缓存, 如已构建)")
        print(f"└── test/")
        print(f"    ├── digit_0/ ... digit_9/  (共 10,000 个 .npy 文件)")
        print(f"    └── phase_test_cache.npz   (缓存, 如已构建)")
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description='MNIST 图像到相位全息图转换管道',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  # 完整流程: 转换 + 生成缓存
  python3 minst_image_to_phase_process.py
  
  # 仅转换，不生成缓存
  python3 minst_image_to_phase_process.py --no-cache
  
  # 自定义参数
  python3 minst_image_to_phase_process.py \\
    -md ./data/minst \\
    -od ./data/minst_phase \\
    -r 256 -i 200
  
  # 仅验证现有缓存
  python3 minst_image_to_phase_process.py --verify-only
        ''')
    
    parser.add_argument('--mnist-dir', '-md', type=str,
                       default='./data/minst',
                       help='MNIST 数据集目录 (默认: ./data/minst)')
    parser.add_argument('--output-dir', '-od', type=str,
                       default='./data/minst_phase',
                       help='输出目录 (默认: ./data/minst_phase)')
    parser.add_argument('--resolution', '-r', type=int, default=256,
                       help='相位处理分辨率 (默认: 256)')
    parser.add_argument('--iterations', '-i', type=int, default=200,
                       help='GS 迭代次数 (默认: 200)')
    parser.add_argument('--no-cache', action='store_true',
                       help='不生成 NPZ 缓存')
    parser.add_argument('--verify-only', action='store_true',
                       help='仅验证现有缓存，不进行处理')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='安静模式')
    
    args = parser.parse_args()
    
    converter = MNISTPhaseConverter(
        args.mnist_dir,
        args.output_dir,
        args.resolution,
        args.iterations,
        verbose=not args.quiet
    )
    
    if args.verify_only:
        # 仅验证缓存
        success = all(converter.verify_cache(split) for split in ['train', 'test'])
    else:
        # 完整处理流程
        success = converter.process_all(build_cache=not args.no_cache)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
