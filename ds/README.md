# DS 数据处理工具说明

本目录包含 MNIST 原始数据 → 相位全息图 NPZ 的预处理工具。

## 文件说明

| 文件 | 说明 |
|------|------|
| `minst_image_to_phase_process.py` | 主处理脚本：MNIST IDX → 相位全息图 NPZ |
| `phase_extractor.py` | GS 相位提取器，供处理脚本调用 |
| `phase_visualizer.py` | 相位图可视化工具（当前项目未主动调用，可独立使用） |

## 数据生产启动命令

默认使用 **64×64** 分辨率、**多进程**并行处理：

```bash
python3 ds/minst_image_to_phase_process.py
```

输出目录：`data/minst_phase/`

### 常用参数

```bash
# 指定输入/输出目录
nohup python3 ds/minst_image_to_phase_process.py \
  --resolution 64
  --mnist-dir ./data/minst \
  --output-dir ./data/minst_phase_64 \
  --worker 8  \
  > ds/minst_phase_64.log 2>&1 &

# 仅验证已生成的 NPZ 文件
python3 ds/minst_image_to_phase_process.py \
  --output-dir ./data/minst_phase \
  --verify-only
```

### 并行方式说明

- **多进程（默认）**：GS 迭代是 CPU 密集型任务，多进程能绕过 Python GIL，充分利用多核 CPU。
- **多线程（`--use-threads`）**：进程启动较慢或环境限制时使用，但 CPU 利用率通常不如多进程。

### 输出结构

```text
data/minst_phase/
├── train/
│   ├── minst_phase_train_01.npz
│   ├── minst_phase_train_02.npz
│   └── ...
└── test/
    └── minst_phase_test_01.npz
```

每个 `.npz` 文件包含：

- `phase`: `(N, resolution, resolution)` float32 相位数据，范围 [-π, π]
- `labels`: `(N,)` int64 标签，取值 0-9

## 分辨率与模型配置对应关系

| 数据分辨率 | 对应模型 `img_size` | 备注 |
|-----------|---------------------|------|
| 64×64     | `img_size: 64`      | 推荐，计算高效 |
| 128×128   | `img_size: 128`     | 平衡选择 |
| 256×256   | `img_size: 256`     | 历史配置，模型较大 |

**注意**：生成数据后，需要同步修改 `config/*.yaml` 中的 `model.img_size` 与 `model.patch_size`，确保与数据分辨率匹配。
