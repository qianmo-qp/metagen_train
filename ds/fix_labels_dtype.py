#!/usr/bin/env python3
"""
一次性脚本：将已生成 NPZ 文件中的 labels 从 uint8 转换为 int64（原地重写）。

用法:
    python3 ds/fix_labels_dtype.py --data-dir ./data/minst_phase_64

phase 数据保持不变，仅修正 labels 的 dtype，使其通过 verify_npz_files 校验。
"""
import argparse
import glob
import os
import numpy as np


def fix_file(npz_path):
    with np.load(npz_path, allow_pickle=False) as data:
        phase = data['phase']
        labels = data['labels']
        old_dtype = labels.dtype
        if labels.dtype == np.int64:
            return old_dtype, False  # 已是 int64，无需处理
        labels = labels.astype(np.int64)
        phase = np.array(phase)  # 复制出 npz 句柄

    # 原地重写（先写临时再替换，避免中途失败损坏原文件）
    # 注意：np.savez 会自动为不以 .npz 结尾的文件名追加 .npz 后缀
    tmp_path = npz_path + '.tmp.npz'
    np.savez(tmp_path, phase=phase, labels=labels)
    os.replace(tmp_path, npz_path)
    return old_dtype, True


def main():
    parser = argparse.ArgumentParser(description='修正 NPZ labels dtype 为 int64')
    parser.add_argument('--data-dir', '-d', type=str, default='./data/minst_phase_64',
                        help='包含 train/test 子目录的数据根目录')
    args = parser.parse_args()

    npz_files = sorted(glob.glob(os.path.join(args.data_dir, '**', '*.npz'), recursive=True))
    npz_files = [f for f in npz_files if '.tmp' not in os.path.basename(f)]
    if not npz_files:
        print(f"❌ 未找到 NPZ 文件: {args.data_dir}")
        return

    print(f"找到 {len(npz_files)} 个 NPZ 文件，开始修正 labels dtype...\n")
    fixed = 0
    for npz_path in npz_files:
        old_dtype, changed = fix_file(npz_path)
        status = f"{old_dtype} -> int64 ✓" if changed else f"已是 int64，跳过"
        print(f"  {os.path.relpath(npz_path, args.data_dir)}: {status}")
        if changed:
            fixed += 1

    print(f"\n✅ 完成，共修正 {fixed} 个文件。")


if __name__ == '__main__':
    main()
