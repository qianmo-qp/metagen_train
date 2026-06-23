"""
MNIST Batch Phase Extraction
批量处理MNIST数据集,提取相位全息图并保存
"""

import numpy as np
import os
import sys
from pathlib import Path
from tqdm import tqdm
import argparse

# 导入相位提取器
from phase_extractor import PhaseExtractor


class MNISTPhaseProcessor:
    """MNIST批量处理"""
    
    def __init__(self, mnist_data_dir, output_dir, resolution=256, iterations=200, verbose=True):
        """
        参数:
            mnist_data_dir: MNIST数据集目录 (包含train-images.idx3-ubyte等)
            output_dir: 输出目录
            resolution: 处理分辨率
            iterations: GS迭代次数
            verbose: 是否打印进度
        """
        self.mnist_data_dir = mnist_data_dir
        self.output_dir = output_dir
        self.resolution = resolution
        self.iterations = iterations
        self.verbose = verbose
        self.extractor = PhaseExtractor(resolution, iterations, verbose=False)
    
    def read_idx_images(self, filename):
        """
        读取MNIST IDX格式图像文件
        
        参数:
            filename: IDX文件路径
        
        返回:
            images: (N, 28, 28) 数组
        """
        with open(filename, 'rb') as f:
            magic = np.frombuffer(f.read(4), dtype='>i4')[0]
            num_images = np.frombuffer(f.read(4), dtype='>i4')[0]
            height = np.frombuffer(f.read(4), dtype='>i4')[0]
            width = np.frombuffer(f.read(4), dtype='>i4')[0]
            
            images = np.frombuffer(f.read(), dtype='>u1').reshape(num_images, height, width)
            return images
    
    def read_idx_labels(self, filename):
        """
        读取MNIST IDX格式标签文件
        
        参数:
            filename: IDX文件路径
        
        返回:
            labels: (N,) 数组
        """
        with open(filename, 'rb') as f:
            magic = np.frombuffer(f.read(4), dtype='>i4')[0]
            num_items = np.frombuffer(f.read(4), dtype='>i4')[0]
            
            labels = np.frombuffer(f.read(), dtype='>u1')
            return labels
    
    def image_to_phase(self, image_28x28):
        """
        将单个MNIST图像转换为相位
        
        参数:
            image_28x28: (28, 28) 灰度图像 [0, 255]
        
        返回:
            phase: (resolution, resolution) 相位 [-π, π]
        """
        from PIL import Image as PILImage
        
        # 转为[0, 1]
        img_array = image_28x28.astype(float) / 255.0
        
        # 放大到处理分辨率
        img_pil = PILImage.fromarray((img_array * 255).astype(np.uint8))
        img_pil = img_pil.resize((self.resolution, self.resolution), PILImage.Resampling.LANCZOS)
        img_array = np.array(img_pil).astype(float) / 255.0
        
        # 增强对比度
        img_array = (img_array - 0.5) * 1.3 + 0.5
        img_array = np.clip(img_array, 0, 1)
        
        # GS算法
        phase, _ = self.extractor.gs_algorithm(img_array)
        
        return phase
    
    def process_dataset(self, split='train'):
        """
        处理数据集 (train或test)
        
        参数:
            split: 'train' 或 'test'
        """
        # 确定文件路径
        if split == 'train':
            images_file = os.path.join(self.mnist_data_dir, 'train-images.idx3-ubyte')
            labels_file = os.path.join(self.mnist_data_dir, 'train-labels.idx1-ubyte')
        elif split == 'test':
            images_file = os.path.join(self.mnist_data_dir, 't10k-images.idx3-ubyte')
            labels_file = os.path.join(self.mnist_data_dir, 't10k-labels.idx1-ubyte')
        else:
            raise ValueError(f"split必须是'train'或'test',得到{split}")
        
        # 检查文件
        if not os.path.exists(images_file):
            print(f"❌ 文件不存在: {images_file}")
            return False
        if not os.path.exists(labels_file):
            print(f"❌ 文件不存在: {labels_file}")
            return False
        
        if self.verbose:
            print(f"\n[{split.upper()}数据集] 开始处理...")
        
        # 读取数据
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
            
            # 保存相位
            digit_dir = os.path.join(output_split_dir, f'digit_{label}')
            phase_filename = f'phase_{processed_count[label]:05d}.npy'
            phase_path = os.path.join(digit_dir, phase_filename)
            
            np.save(phase_path, phase)
            
            processed_count[label] += 1
        
        # 统计信息
        if self.verbose:
            print(f"\n✓ {split.upper()}处理完成!")
            for digit in range(10):
                print(f"  数字{digit}: {processed_count[digit]} 个")
        
        return True
    
    def process_all(self):
        """处理全部数据集"""
        print("=" * 60)
        print("MNIST批量相位提取")
        print("=" * 60)
        
        # 创建主输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        if self.verbose:
            print(f"\n输入目录: {self.mnist_data_dir}")
            print(f"输出目录: {self.output_dir}")
            print(f"分辨率: {self.resolution}×{self.resolution}")
            print(f"迭代次数: {self.iterations}")
        
        # 处理训练集
        train_ok = self.process_dataset('train')
        
        # 处理测试集
        test_ok = self.process_dataset('test')
        
        if train_ok and test_ok:
            print("\n" + "=" * 60)
            print("✓ 所有数据集处理完成!")
            print("=" * 60)
            print(f"\n目录结构:")
            print(f"{self.output_dir}/")
            print(f"├── train/")
            print(f"│   ├── digit_0/")
            print(f"│   │   ├── phase_00000.npy")
            print(f"│   │   ├── phase_00001.npy")
            print(f"│   │   └── ...")
            print(f"│   ├── digit_1/")
            print(f"│   └── ...")
            print(f"└── test/")
            print(f"    └── ...")
        else:
            print("\n❌ 处理失败!")
            return False
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description='MNIST批量相位提取',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python3 mnist_image_process.py                    # 使用默认路径
  python3 mnist_image_process.py -r 128 -i 100    # 自定义分辨率和迭代次数
  python3 mnist_image_process.py -md ./data/minst -od ./data/phase_output
        ''')
    parser.add_argument('--mnist-dir', '-md', type=str, 
                       default='./data/minst',
                       help='MNIST数据集目录')
    parser.add_argument('--output-dir', '-od', type=str,
                       default='./data/minst_phase',
                       help='输出目录')
    parser.add_argument('--resolution', '-r', type=int, default=256,
                       help='处理分辨率 (默认: 256)')
    parser.add_argument('--iterations', '-i', type=int, default=200,
                       help='GS迭代次数 (默认: 200)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='安静模式,不显示进度')
    
    args = parser.parse_args()
    
    # 创建处理器
    processor = MNISTPhaseProcessor(
        args.mnist_dir,
        args.output_dir,
        args.resolution,
        args.iterations,
        verbose=not args.quiet
    )
    
    # 处理数据集
    success = processor.process_all()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
