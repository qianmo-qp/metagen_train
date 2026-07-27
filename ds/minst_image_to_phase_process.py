"""
MNIST Image to Phase Hologram Processing Pipeline (Optimized)
MNIST 图像 → 相位全息图 NPZ 文件（直接生成，无中间文件）

优化特性：
1. 无中间 .npy 文件 - 直接生成 NPZ
2. 按 10,000 个数据一个文件分割
3. float32 精度（无损存储）
4. 多进程处理（默认 8 个进程，比多线程更适合 CPU 密集 GS 迭代）
5. 自动编号（train: 01-06, test: 01）
6. 默认分辨率 64×64（更匹配 MNIST 原始信息尺度）
"""

import numpy as np
import os
import sys
from pathlib import Path
from tqdm import tqdm
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# 导入相位提取器
from phase_extractor import PhaseExtractor


def _image_to_phase(image_28x28, extractor):
    """
    将单个 MNIST 图像转换为相位全息图

    参数:
        image_28x28: (28, 28) 灰度图像，像素值 [0, 255]
        extractor: PhaseExtractor 实例

    返回:
        phase: (resolution, resolution) 相位全息图，范围 [-π, π]
    """
    from PIL import Image as PILImage

    # 1. 归一化到 [0, 1]
    img_array = image_28x28.astype(float) / 255.0

    # 2. 上采样到处理分辨率
    img_pil = PILImage.fromarray((img_array * 255).astype(np.uint8))
    img_pil = img_pil.resize((extractor.N, extractor.N), PILImage.Resampling.LANCZOS)
    img_array = np.array(img_pil).astype(float) / 255.0

    # 3. 对比度增强
    img_array = (img_array - 0.5) * 1.3 + 0.5
    img_array = np.clip(img_array, 0, 1)

    # 4. GS 算法提取相位
    phase, _ = extractor.gs_algorithm(img_array)

    return phase


def _process_batch_worker(args):
    """
    多进程 worker：处理一个 batch 的图像

    参数:
        args: (images_slice, labels_slice, start_idx, resolution, iterations)

    返回:
        (start_idx, phase_data, labels_slice)
    """
    images_slice, labels_slice, start_idx, resolution, iterations = args

    extractor = PhaseExtractor(resolution, iterations, verbose=False)
    batch_size = len(images_slice)
    phase_data = np.zeros((batch_size, resolution, resolution), dtype=np.float32)

    for idx in range(batch_size):
        phase_data[idx] = _image_to_phase(images_slice[idx], extractor).astype(np.float32)

    return start_idx, phase_data, labels_slice


class OptimizedPhaseConverter:
    """优化版本的相位转换处理器"""

    def __init__(self, mnist_data_dir, output_dir, resolution=64, iterations=200,
                 num_workers=8, chunk_size=2500, use_threads=False, verbose=True):
        """
        参数:
            mnist_data_dir: MNIST 数据集目录
            output_dir: 输出目录
            resolution: 相位处理分辨率（默认 64，推荐 64/128/256）
            iterations: GS 迭代次数
            num_workers: 并行 worker 数量
            chunk_size: 每个 NPZ 文件包含的样本数（建议设为 num_workers 的整数倍以提升并行效率）
            use_threads: 是否使用多线程（默认 False，使用多进程）。
                         GS 迭代是 CPU 密集型任务，多进程通常更快；仅当进程启动开销过大时启用多线程。
            verbose: 是否打印详细信息
        """
        self.mnist_data_dir = mnist_data_dir
        self.output_dir = output_dir
        self.resolution = resolution
        self.iterations = iterations
        self.num_workers = num_workers
        self.chunk_size = chunk_size
        self.use_threads = use_threads
        self.verbose = verbose

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

    # ======================== 多进程/多线程批处理 ========================

    def process_dataset_parallel(self, split='train'):
        """
        使用多进程/多线程并行处理数据集，直接生成 NPZ 文件

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
            print(f"  使用 {self.num_workers} 个 {'线程' if self.use_threads else '进程'} 处理\n")

        # 准备任务参数：每个 worker 只拿到自己的 slice，避免重复序列化整份数据
        tasks = []
        for chunk_id in range(num_chunks):
            start_idx = chunk_id * self.chunk_size
            end_idx = min((chunk_id + 1) * self.chunk_size, total_samples)
            tasks.append((
                images[start_idx:end_idx],
                labels[start_idx:end_idx],
                start_idx,
                self.resolution,
                self.iterations,
            ))

        # 收集每个 batch 的结果并按 start_idx 排序
        results = [None] * num_chunks

        ExecutorClass = ThreadPoolExecutor if self.use_threads else ProcessPoolExecutor
        with ExecutorClass(max_workers=self.num_workers) as executor:
            futures = {executor.submit(_process_batch_worker, task): i for i, task in enumerate(tasks)}

            pbar = tqdm(total=num_chunks, desc=f"处理{split}数据集", disable=not self.verbose)
            for future in as_completed(futures):
                chunk_id = futures[future]
                try:
                    start_idx, phase_data, labels_data = future.result()
                    results[chunk_id] = (start_idx, phase_data, labels_data)
                    pbar.update(1)
                except Exception as e:
                    pbar.close()
                    print(f"❌ 处理 chunk {chunk_id} 失败: {e}")
                    return False
            pbar.close()

        # 保存 NPZ 文件
        for chunk_id, (start_idx, phase_data, labels_data) in enumerate(results):
            if phase_data is None:
                print(f"❌ chunk {chunk_id} 结果缺失")
                return False

            chunk_num = chunk_id + 1
            chunk_filename = f'minst_phase_{split}_{chunk_num:02d}.npz'
            chunk_path = os.path.join(output_split_dir, chunk_filename)

            if self.verbose:
                print(f"  正在保存: {chunk_filename} ({phase_data.nbytes / (1024**2):.1f} MB)...", end='', flush=True)

            np.savez(
                chunk_path,
                phase=phase_data,
                labels=np.asarray(labels_data, dtype=np.int64)
            )

            if self.verbose:
                print(f" ✓ 完成")

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
                    f"Phase 分辨率错误: {phase.shape[1:]}，期望 {(self.resolution, self.resolution)}"
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
            print(f"  并行方式: {'多线程' if self.use_threads else '多进程'}")
            print(f"  Worker 数量: {self.num_workers}")
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
        print(f"│   ├── minst_phase_train_01.npz  (样本 0-{self.chunk_size-1})")
        print(f"│   ├── ...")
        print(f"│   └── minst_phase_train_{(60000 + self.chunk_size - 1) // self.chunk_size:02d}.npz")
        print(f"└── test/")
        print(f"    └── minst_phase_test_01.npz")
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
  # 完整处理（默认 64×64，多进程）
  python3 minst_image_to_phase_process.py

  # 自定义参数：128×128，16 进程
  python3 minst_image_to_phase_process.py \\
    -md ./data/minst \\
    -od ./data/minst_phase \\
    -r 128 -i 200 \\
    -w 16 --chunk-size 2500

  # 仅验证
  python3 minst_image_to_phase_process.py --verify-only
        ''')

    parser.add_argument('--mnist-dir', '-md', type=str,
                       default='./data/minst',
                       help='MNIST 数据集目录 (默认: ./data/minst)')
    parser.add_argument('--output-dir', '-od', type=str,
                       default='./data/minst_phase',
                       help='输出目录 (默认: ./data/minst_phase)')
    parser.add_argument('--resolution', '-r', type=int, default=64,
                       help='相位处理分辨率 (默认: 64，可选 64/128/256)')
    parser.add_argument('--iterations', '-i', type=int, default=200,
                       help='GS 迭代次数 (默认: 200)')
    parser.add_argument('--workers', '-w', type=int, default=8,
                       help='并行 worker 数量 (默认: 8)')
    parser.add_argument('--chunk-size', type=int, default=2500,
                       help='每个 worker 处理的样本数 (默认: 2500，建议为 workers 的整数倍)')
    parser.add_argument('--use-threads', action='store_true',
                       help='使用多线程而非多进程（默认使用多进程，更适合 CPU 密集 GS 迭代）')
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
        use_threads=args.use_threads,
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
