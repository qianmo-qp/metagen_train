"""
Phase Visualizer: Phase Map (phase_map.npy) → PNG Images
将相位全息图转换为可视化图像 (彩色、灰度、恢复)
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import argparse


class PhaseVisualizer:
    """可视化相位全息图"""
    
    def __init__(self, verbose=True):
        """
        参数:
            verbose: 是否打印进度
        """
        self.verbose = verbose
    
    def load_phase(self, phase_path):
        """
        加载相位数据
        
        参数:
            phase_path: .npy文件路径
        
        返回:
            phase: 相位数组 [-π, π]
        """
        phase = np.load(phase_path)
        
        if self.verbose:
            print(f"[加载] {phase_path}")
            print(f"  形状: {phase.shape}")
            print(f"  范围: [{phase.min():.4f}, {phase.max():.4f}]")
        
        return phase
    
    def phase_to_color(self, phase, output_path=None):
        """
        相位 → 彩色图 (twilight色图)
        
        参数:
            phase: 相位数组
            output_path: 保存路径 (可选)
        
        返回:
            color_img: 彩色图像
        """
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
        ax.set_title('Phase Hologram (Color)')
        ax.axis('off')
        plt.colorbar(im, ax=ax, label='Phase (rad)')
        
        if output_path:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            
            if self.verbose:
                print(f"[保存] {output_path} (彩色)")
        
        plt.close()
    
    def phase_to_16bit_gray(self, phase, output_path=None):
        """
        相位 → 16bit灰度图 (用于SLM)
        
        参数:
            phase: 相位数组
            output_path: 保存路径 (可选)
        
        返回:
            phase_16bit: 16bit数组
        """
        # 映射 [-π, π] → [0, 65535]
        phase_16bit = ((phase + np.pi) / (2 * np.pi) * 65535).astype(np.uint16)
        
        if output_path:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            
            # 使用PIL保存以保持16bit精度
            img = Image.fromarray(phase_16bit)
            img.save(output_path)
            
            if self.verbose:
                print(f"[保存] {output_path} (16bit灰度)")
        
        return phase_16bit
    
    def recover_image_from_phase(self, phase, refinement_iterations=5):
        """
        从相位恢复图像 (FFT重建)
        
        参数:
            phase: 相位数组
            refinement_iterations: 细化迭代次数
        
        返回:
            recovered: 恢复的图像 [0, 1]
        """
        # 简单恢复
        u1 = np.exp(1j * phase)
        u2 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u1)))
        recovered = np.abs(u2)
        recovered = recovered / np.max(recovered)
        
        if self.verbose:
            print(f"\n[恢复] FFT反向变换")
            print(f"  恢复图像范围: [{recovered.min():.4f}, {recovered.max():.4f}]")
        
        return recovered
    
    def phase_to_recovered_png(self, phase, output_path=None):
        """
        相位 → 恢复图像PNG
        
        参数:
            phase: 相位数组
            output_path: 保存路径 (可选)
        
        返回:
            recovered: 恢复的图像
        """
        recovered = self.recover_image_from_phase(phase)
        
        if output_path:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            
            # 转为uint8并保存
            recovered_uint8 = (recovered * 255).astype(np.uint8)
            img = Image.fromarray(recovered_uint8)
            img.save(output_path)
            
            if self.verbose:
                print(f"[保存] {output_path} (恢复图像)")
        
        return recovered
    
    def save_all_visualizations(self, phase, output_dir, base_name='phase'):
        """
        保存所有可视化格式
        
        参数:
            phase: 相位数组
            output_dir: 输出目录
            base_name: 文件基名
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 彩色图
        color_path = os.path.join(output_dir, f'{base_name}_Color.png')
        self.phase_to_color(phase, color_path)
        
        # 16bit灰度
        gray_path = os.path.join(output_dir, f'{base_name}_16bit.png')
        self.phase_to_16bit_gray(phase, gray_path)
        
        # 恢复图像
        recovered_path = os.path.join(output_dir, f'{base_name}_Recovered.png')
        self.phase_to_recovered_png(phase, recovered_path)
        
        if self.verbose:
            print(f"\n✓ 所有可视化已保存到: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='可视化相位全息图')
    parser.add_argument('phase_file', type=str, help='输入相位文件路径 (.npy)')
    parser.add_argument('--output-dir', '-od', type=str, default=None,
                       help='输出目录 (默认: 同名目录)')
    parser.add_argument('--format', '-f', type=str, choices=['all', 'color', '16bit', 'recovered'],
                       default='all', help='输出格式 (默认: all)')
    parser.add_argument('--color', action='store_true', help='仅输出彩色图')
    parser.add_argument('--16bit', dest='bit16', action='store_true', help='仅输出16bit灰度')
    parser.add_argument('--recovered', action='store_true', help='仅输出恢复图像')
    parser.add_argument('--quiet', '-q', action='store_true', help='安静模式')
    
    args = parser.parse_args()
    
    # 确定输出目录
    if args.output_dir is None:
        base_name = os.path.splitext(args.phase_file)[0]
        args.output_dir = f"{base_name}_visualizations"
    
    # 加载相位
    visualizer = PhaseVisualizer(verbose=not args.quiet)
    phase = visualizer.load_phase(args.phase_file)
    
    # 确定输出格式
    if args.color or args.bit16 or args.recovered:
        # 单一格式
        if args.color:
            output_path = os.path.join(args.output_dir, 'phase_Color.png')
            visualizer.phase_to_color(phase, output_path)
        if args.bit16:
            output_path = os.path.join(args.output_dir, 'phase_16bit.png')
            visualizer.phase_to_16bit_gray(phase, output_path)
        if args.recovered:
            output_path = os.path.join(args.output_dir, 'phase_Recovered.png')
            visualizer.phase_to_recovered_png(phase, output_path)
    else:
        # 全部格式
        visualizer.save_all_visualizations(phase, args.output_dir, 'phase')
    
    print(f"\n✓ 完成!")


if __name__ == "__main__":
    main()
