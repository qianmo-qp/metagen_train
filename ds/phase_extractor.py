"""
Phase Extractor: PNG Image → Phase Map (phase_map.npy)
从PNG图像提取相位全息图,保存为npy格式
"""

import numpy as np
from PIL import Image
import os
import argparse


class PhaseExtractor:
    """从图像提取相位全息图"""
    
    def __init__(self, resolution=256, iterations=200, verbose=True, smoothness=0.0):
        """
        参数:
            resolution: 处理分辨率 (推荐256)
            iterations: GS迭代次数
            verbose: 是否打印进度
            smoothness: 相位平滑约束强度 [0, 1]。0=关闭（原始 GS，相位高熵近白噪声）；
                        推荐 0.3~0.6。越大相位越平滑（空间熵越低，越利于扩散模型学习），
                        但重建振幅保真度会下降。
        """
        self.N = resolution
        self.iterations = iterations
        self.verbose = verbose
        self.smoothness = float(smoothness)
        self._smooth_kernel = None
        if self.smoothness > 0:
            # 高斯核标准差随分辨率缩放（以 64 为基准，×2 使约束更有效）
            sigma = max(0.8, 2.0 * self.N / 64.0)
            self._smooth_kernel = self._make_circular_gaussian_kernel(sigma)
    
    def _make_circular_gaussian_kernel(self, sigma):
        """构造 N×N 循环高斯平滑核（频域相乘，支持相位环绕边界）。"""
        N = self.N
        ax = np.arange(N)
        # 循环距离坐标
        dx = np.minimum(ax, N - ax)
        g1 = np.exp(-(dx ** 2) / (2.0 * sigma ** 2))
        kernel = np.outer(g1, g1)
        kernel /= kernel.sum()
        # 预计算 FFT；fftshift 使核原点居中以便与相位直接相乘
        return np.fft.fft2(np.fft.ifftshift(kernel))
    
    def _smooth_phase(self, phase):
        """对相位做循环高斯平滑（处理 ±π 环绕），返回平滑后相位 [-π, π]。"""
        z = np.exp(1j * phase)  # 映射到单位圆，避免 ±π 边界伪影
        z_smooth = np.fft.ifft2(self._smooth_kernel * np.fft.fft2(z))
        return np.angle(z_smooth)
    
    def load_and_preprocess(self, image_path, target_size=None):
        """
        加载并预处理图像
        
        参数:
            image_path: 图像路径 (PNG/JPG)
            target_size: 目标尺寸 (默认为self.N)
        
        返回:
            img_array: 预处理后的图像 [0, 1]
        """
        if target_size is None:
            target_size = self.N
        
        # 加载图像
        img = Image.open(image_path).convert('L')  # 灰度
        
        if self.verbose:
            print(f"[加载] {image_path}")
            print(f"  原始尺寸: {img.size}")
        
        # 调整到目标尺寸
        img = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
        img_array = np.array(img).astype(float) / 255.0  # [0, 1]
        
        if self.verbose:
            print(f"  缩放后: {target_size}×{target_size}")
            print(f"  像素范围: [{img_array.min():.4f}, {img_array.max():.4f}]")
        
        # 增强对比度
        img_array = (img_array - 0.5) * 1.3 + 0.5
        img_array = np.clip(img_array, 0, 1)
        
        if self.verbose:
            print(f"  增强后范围: [{img_array.min():.4f}, {img_array.max():.4f}]")
        
        return img_array
    
    def gs_algorithm(self, target_amp):
        """
        Gerchberg-Saxton算法提取相位
        
        参数:
            target_amp: 目标振幅分布 (N×N)
        
        返回:
            phase_slm: 恢复的相位 [-π, π]
            mse_history: MSE收敛历史
        """
        # 初始化
        amp_slm = np.ones((self.N, self.N))
        phase_slm = np.random.rand(self.N, self.N) * 2 * np.pi
        
        # 能量归一化
        energy_slm = np.sum(amp_slm**2)
        energy_target = np.sum(target_amp**2)
        target_scaled = target_amp * np.sqrt(energy_slm / energy_target)
        
        mse_history = []
        
        if self.verbose:
            print(f"\n[GS算法] 开始迭代 (N={self.N}×{self.N}, 迭代={self.iterations})")
        
        for it in range(self.iterations):
            # 正向: SLM → 像面
            u1 = amp_slm * np.exp(1j * phase_slm)
            u2 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(u1)))
            amp_image = np.abs(u2)
            phase_image = np.angle(u2)
            
            # 误差
            mse = np.mean((amp_image - target_scaled)**2)
            mse_history.append(mse)
            
            if self.verbose and (it+1) % max(1, self.iterations//10) == 0:
                print(f"  迭代 {it+1:3d}/{self.iterations}, MSE = {mse:.8f}")
            
            # 施加约束
            u2_new = target_scaled * np.exp(1j * phase_image)
            
            # 反向: 像面 → SLM
            u1_new = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(u2_new)))
            phase_slm = np.angle(u1_new)
            
            # 平滑约束：将相位向其平滑版本混合，降低空间熵（抑制高频随机性）
            if self.smoothness > 0:
                phase_smooth = self._smooth_phase(phase_slm)
                # 在单位圆上混合后再取角度，正确处理 ±π 环绕
                z = (1.0 - self.smoothness) * np.exp(1j * phase_slm) + \
                    self.smoothness * np.exp(1j * phase_smooth)
                phase_slm = np.angle(z)
        
        if self.verbose:
            print(f"✓ GS完成!")
            print(f"  最终MSE: {mse_history[-1]:.8f}")
            print(f"  相位范围: [{phase_slm.min():.4f}, {phase_slm.max():.4f}]")
        
        return phase_slm, mse_history
    
    def extract_phase(self, image_path):
        """
        完整流程: 图像 → 相位
        
        参数:
            image_path: 输入图像路径
        
        返回:
            phase: 相位全息图 [-π, π]
        """
        # 加载预处理
        target_amp = self.load_and_preprocess(image_path)
        
        # GS算法
        phase, mse_history = self.gs_algorithm(target_amp)
        
        return phase, target_amp, mse_history
    
    def save_phase(self, phase, output_path):
        """
        保存相位为npy格式
        
        参数:
            phase: 相位数据
            output_path: 输出路径 (.npy)
        """
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        np.save(output_path, phase)
        
        if self.verbose:
            print(f"\n[保存] {output_path}")
            print(f"  形状: {phase.shape}")
            print(f"  范围: [{phase.min():.4f}, {phase.max():.4f}]")


def main():
    parser = argparse.ArgumentParser(description='从PNG图像提取相位全息图')
    parser.add_argument('input_image', type=str, help='输入图像路径 (PNG/JPG)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='输出npy路径 (默认: 同名.npy)')
    parser.add_argument('--resolution', '-r', type=int, default=256,
                       help='处理分辨率 (默认: 256)')
    parser.add_argument('--iterations', '-i', type=int, default=200,
                       help='GS迭代次数 (默认: 200)')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='安静模式,不打印进度')
    
    args = parser.parse_args()
    
    # 确定输出路径
    if args.output is None:
        base_name = os.path.splitext(args.input_image)[0]
        args.output = f"{base_name}_phase_map.npy"
    
    # 执行提取
    extractor = PhaseExtractor(
        resolution=args.resolution,
        iterations=args.iterations,
        verbose=not args.quiet
    )
    
    phase, target_amp, mse_history = extractor.extract_phase(args.input_image)
    extractor.save_phase(phase, args.output)
    
    print(f"\n✓ 完成! 相位已保存到: {args.output}")


if __name__ == "__main__":
    main()
