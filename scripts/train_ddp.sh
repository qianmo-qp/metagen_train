#!/bin/bash
# DDP Training Launch Script for Multi-GPU Training
# Usage: ./scripts/train_ddp.sh [num_gpus]

# Default to 4 GPUs if not specified
NUM_GPUS=${1:-4}

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
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
