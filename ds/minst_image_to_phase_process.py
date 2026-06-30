"""
MNIST Image to Phase Hologram Processing Pipeline (Optimized)
MNIST 图像 → 相位全息图 NPZ 文件（直接生成，无中间文件）

优化特性：
1. 无中间 .npy 文件 - 直接生成 NPZ
2. 按 10,000 个数据一个文件分割
3. float32 精度（无损存储）
4. 多线程处理（8 个线程）
5. 自动编号（train: 01-06, test: 01）
"""

import numpy as np
import os
import sys
from pathlib import Path
from tqdm import tqdm
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 导入相位提取器
from phase_extractor import PhaseExtractor


class OptimizedPhaseConverter:
    """优化版本的相位转换处理器"""
    
    def __init__(self, mnist_data_dir, output_dir, resolution=256, iterations=200, 
                 num_workers=8, chunk_size=10000, verbose=True):
        """
        参数:
            mnist_data_dir: MNIST 数据集目录
            output_dir: 输出目录
            resolution: 相位处理分辨率
            iterations: GS 迭代次数
            num_workers: 多线程数量
            chunk_size: 每个 NPZ 文件包含的样本数
            verbose: 是否打印详细信息
        """
        self.mnist_data_dir = mnist_data_dir
        self.output_dir = output_dir
        self.resolution = resolution
        self.iterations = iterations
        self.num_workers = num_workers
        self.chunk_size = chunk_size
        self.verbose = verbose
        self.extractor = PhaseExtractor(resolution, iterations, verbose=False)
        self.lock = threading.Lock()
    
    # ======================== IDX 格式读取 ========================
    
    def read_idx_images(self, filename):
        """
        读取 MNIST IDX 格式图像文件
        
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
    
    # ======================== 多线程批处理 ========================
    
    def process_batch(self, images, labels, start_idx, end_idx):
        """
        多线程处理一个批次的图像
        
        参数:
            images: MNIST 图像数组
            labels: MNIST 标签数组
            start_idx: 起始索引
            end_idx: 结束索引
        
        返回:
            phase_data: 相位数据 (float32)
            labels_data: 标签数据
        """
        batch_size = end_idx - start_idx
        phase_data = np.zeros((batch_size, self.resolution, self.resolution), dtype=np.float32)
        labels_data = np.zeros(batch_size, dtype=np.int64)
        
        # 实时反馈处理进度
        import time
        start_time = time.time()
        log_interval = max(1, batch_size // 10)  # 每处理 10% 输出一次日志
        
        for idx in range(batch_size):
            image_idx = start_idx + idx
            phase_data[idx] = self.image_to_phase(images[image_idx]).astype(np.float32)
            labels_data[idx] = labels[image_idx]
            
            # 每处理 log_interval 个样本输出一次进度
            if (idx + 1) % log_interval == 0 or idx + 1 == batch_size:
                elapsed = time.time() - start_time
                rate = (idx + 1) / elapsed if elapsed > 0 else 0
                progress_pct = (idx + 1) / batch_size * 100
                print(f"    [batch {start_idx}-{end_idx}] 处理进度: {idx + 1}/{batch_size} ({progress_pct:.1f}%) "
                      f"[{elapsed:.1f}s, {rate:.1f} img/s]")
        
        return phase_data, labels_data
    
    def process_dataset_parallel(self, split='train'):
        """
        使用多线程并行处理数据集，直接生成 NPZ 文件
        
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
        
        # 计算需要的 NPZ 文件数量和编号
        total_samples = len(images)
        num_chunks = (total_samples + self.chunk_size - 1) // self.chunk_size
        
        if self.verbose:
            print(f"  总样本数: {total_samples}")
            print(f"  分块大小: {self.chunk_size}")
            print(f"  NPZ 文件数: {num_chunks}")
            print(f"  使用 {self.num_workers} 个线程处理\n")
        
        # 多线程处理每个 chunk
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {}
            
            # 提交所有任务
            for chunk_id in range(num_chunks):
                start_idx = chunk_id * self.chunk_size
                end_idx = min((chunk_id + 1) * self.chunk_size, total_samples)
                
                future = executor.submit(
                    self.process_batch,
                    images, labels, start_idx, end_idx
                )
                futures[future] = (chunk_id, start_idx, end_idx)
            
            # 处理完成的任务并保存 NPZ 文件
            pbar = tqdm(total=num_chunks, desc=f"处理{split}数据集", disable=not self.verbose)
            
            for future in as_completed(futures):
                chunk_id, start_idx, end_idx = futures[future]
                
                try:
                    phase_data, labels_data = future.result()
                    
                    # 生成编号（01-06 for train, 01 for test）
                    chunk_num = chunk_id + 1
                    chunk_filename = f'minst_phase_{split}_{chunk_num:02d}.npz'
                    chunk_path = os.path.join(output_split_dir, chunk_filename)
                    
                    # 保存 NPZ 文件
                    if self.verbose:
                        print(f"  正在保存: {chunk_filename} ({phase_data.nbytes / (1024**2):.1f} MB)...", end='', flush=True)
                    
                    np.savez(
                        chunk_path,
                        phase=phase_data,
                        labels=labels_data
                    )
                    
                    if self.verbose:
                        print(f" ✓ 完成")
                    
                    pbar.update(1)
                
                except Exception as e:
                    print(f"❌ 处理 chunk {chunk_id} 失败: {e}")
                    return False
            
            pbar.close()
        
        if self.verbose:
            print(f"✓ {split.upper()} 处理完成!")
            print(f"  生成 {num_chunks} 个 NPZ 文件")
        
        return True
    
    # ======================== 验证和统计 ========================
    
    def verify_npz_files(self, split='train'):
        """
        验证生成的 NPZ 文件
        
        参数:
            split: 'train' 或 'test'
        
        返回:
            success: 是否验证成功
        """
        split_dir = os.path.join(self.output_dir, split)
        
        if not os.path.exists(split_dir):
            print(f"❌ 目录不存在: {split_dir}")
            return False
        
        if self.verbose:
            print(f"\n[验证] {split} NPZ 文件...")
        
        npz_files = sorted([f for f in os.listdir(split_dir) if f.endswith('.npz')])
        
        if not npz_files:
            print(f"❌ 未找到 NPZ 文件在: {split_dir}")
            return False
        
        total_samples = 0
        all_labels = []
        
        try:
            for npz_file in npz_files:
                npz_path = os.path.join(split_dir, npz_file)
                
                # 加载并验证
                data = np.load(npz_path, allow_pickle=False)
                phase = data['phase']
                labels = data['labels']
                
                # 检查形状和类型
                assert phase.ndim == 3, f"Phase 维度错误: {phase.ndim}"
                assert phase.shape[1:] == (self.resolution, self.resolution), \
                    f"Phase 分辨率错误: {phase.shape[1:]}"
                assert phase.dtype == np.float32, f"Phase dtype 错误: {phase.dtype}"
                
                assert labels.ndim == 1, f"Labels 维度错误: {labels.ndim}"
                assert len(labels) == len(phase), f"样本数不匹配"
                assert labels.dtype == np.int64, f"Labels dtype 错误: {labels.dtype}"
                
                total_samples += len(phase)
                all_labels.extend(labels.tolist())
        
        except Exception as e:
            print(f"❌ 验证失败: {e}")
            return False
        
        if self.verbose:
            print(f"✓ NPZ 文件验证通过:")
            print(f"  文件数: {len(npz_files)}")
            print(f"  总样本数: {total_samples}")
            
            # 显示标签分布
            unique, counts = np.unique(all_labels, return_counts=True)
            print(f"  标签分布:")
            for label, count in zip(unique, counts):
                print(f"    数字 {label}: {count:6d} 个")
            
            # 显示文件大小
            total_size_mb = sum(os.path.getsize(os.path.join(split_dir, f)) 
                              for f in npz_files) / (1024 * 1024)
            print(f"  总文件大小: {total_size_mb:.2f} MB")
        
        return True
    
    # ======================== 完整流程 ========================
    
    def process_all(self):
        """
        完整的处理流程：MNIST IDX → NPZ 文件（直接，无中间文件）
        
        返回:
            success: 是否全部成功
        """
        print("=" * 70)
        print("MNIST 图像 → 相位全息图 NPZ 处理管道（优化版）")
        print("=" * 70)
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        if self.verbose:
            print(f"\n配置:")
            print(f"  MNIST 目录: {self.mnist_data_dir}")
            print(f"  输出目录: {self.output_dir}")
            print(f"  相位分辨率: {self.resolution}×{self.resolution}")
            print(f"  GS 迭代次数: {self.iterations}")
            print(f"  多线程数: {self.num_workers}")
            print(f"  文件分块大小: {self.chunk_size} 样本/文件")
            print(f"  数据精度: float32 (无损)")
        
        # 处理训练集和测试集
        train_ok = self.process_dataset_parallel('train')
        test_ok = self.process_dataset_parallel('test')
        
        if not (train_ok and test_ok):
            print("\n❌ 数据集处理失败!")
            return False
        
        print("\n" + "=" * 70)
        print("✓ 所有数据集处理完成!")
        print("=" * 70)
        
        # 验证所有文件
        print("\n" + "=" * 70)
        print("验证生成的 NPZ 文件...")
        print("=" * 70)
        
        train_verify = self.verify_npz_files('train')
        test_verify = self.verify_npz_files('test')
        
        if not (train_verify and test_verify):
            print("\n⚠ 验证失败!")
            return False
        
        print("\n" + "=" * 70)
        print("✓ 完整处理流程完成!")
        print("=" * 70)
        print(f"\n生成的文件结构:")
        print(f"{self.output_dir}/")
        print(f"├── train/")
        print(f"│   ├── minst_phase_train_01.npz  (样本 0-9999)")
        print(f"│   ├── minst_phase_train_02.npz  (样本 10000-19999)")
        print(f"│   ├── minst_phase_train_03.npz  (样本 20000-29999)")
        print(f"│   ├── minst_phase_train_04.npz  (样本 30000-39999)")
        print(f"│   ├── minst_phase_train_05.npz  (样本 40000-49999)")
        print(f"│   └── minst_phase_train_06.npz  (样本 50000-59999)")
        print(f"└── test/")
        print(f"    └── minst_phase_test_01.npz   (样本 0-9999)")
        print(f"\n每个 NPZ 文件包含:")
        print(f"  - 'phase': (N, {self.resolution}, {self.resolution}) float32 相位数据")
        print(f"  - 'labels': (N,) int64 标签 (0-9)")
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description='MNIST 图像到相位全息图转换 (优化版)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  # 完整处理（默认参数）
  python3 minst_image_to_phase_process.py
  
  # 自定义参数
  python3 minst_image_to_phase_process.py \\
    -md ./data/minst \\
    -od ./data/minst_phase \\
    -r 256 -i 200 \\
    -w 16 --chunk-size 5000
  
  # 仅验证
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
    parser.add_argument('--workers', '-w', type=int, default=8,
                       help='多线程数量 (默认: 8)')
    parser.add_argument('--chunk-size', type=int, default=10000,
                       help='每个 NPZ 文件的样本数 (默认: 10000)')
    parser.add_argument('--verify-only', action='store_true',
                       help='仅验证现有 NPZ 文件')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='安静模式')
    
    args = parser.parse_args()
    
    converter = OptimizedPhaseConverter(
        args.mnist_dir,
        args.output_dir,
        args.resolution,
        args.iterations,
        args.workers,
        args.chunk_size,
        verbose=not args.quiet
    )
    
    if args.verify_only:
        # 仅验证
        success = all(converter.verify_npz_files(split) for split in ['train', 'test'])
    else:
        # 完整处理
        success = converter.process_all()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
