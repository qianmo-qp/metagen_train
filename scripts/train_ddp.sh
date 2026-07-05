#!/bin/bash
# DDP Training Launch Script for Multi-GPU Training
# Usage: ./scripts/train_ddp.sh [num_gpus]

# Detect available GPUs
AVAILABLE_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
if [ "$AVAILABLE_GPUS" -eq 0 ]; then
    echo "❌ No GPUs detected!"
    exit 1
fi

# Default to all available GPUs
NUM_GPUS=${1:-$AVAILABLE_GPUS}

# Validate: don't exceed available GPUs
if [ "$NUM_GPUS" -gt "$AVAILABLE_GPUS" ]; then
    echo "⚠️  Requested $NUM_GPUS GPUs but only $AVAILABLE_GPUS available. Using $AVAILABLE_GPUS."
    NUM_GPUS=$AVAILABLE_GPUS
fi

# Dynamically set CUDA_VISIBLE_DEVICES
CUDA_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
export OMP_NUM_THREADS=1

# Get timestamp for log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="outputs/train_ddp_${TIMESTAMP}.log"

# Create outputs directory
mkdir -p outputs

echo "=========================================="
echo "DDP Training Launch"
echo "=========================================="
echo "Num GPUs: $NUM_GPUS"
echo "Log file: $LOG_FILE"
echo "=========================================="

# Launch with torchrun (recommended over torch.distributed.launch)
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    train.py 2>&1 | tee $LOG_FILE

echo "=========================================="
echo "Training complete!"
echo "Log saved to: $LOG_FILE"
echo "=========================================="
