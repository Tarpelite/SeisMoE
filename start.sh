#!/bin/bash

# 设置变量
TOTAL_WORKERS=32
ORIGINAL_CACHE_ROOT="/mnt/data/tianyu/seisbench_cache"
CACHE_DIR="/mnt/data/tianyu/worker_caches"  # worker缓存文件目录
OUTPUT_DIR="./output"
STATUS_DIR="./status"
LOG_DIR="./logs"
FINAL_OUTPUT="STEAD_emd_final.hdf5"

# 创建目录
mkdir -p $CACHE_DIR $OUTPUT_DIR $STATUS_DIR $LOG_DIR

echo "Starting optimized STEAD EMD processing..."
echo "Total workers: $TOTAL_WORKERS"

# 第一步：预处理数据
echo "Step 1: Preprocessing data..."
# python preprocess_data.py --original-cache-root $ORIGINAL_CACHE_ROOT \
#                          --output-dir $CACHE_DIR \
#                          --total-workers $TOTAL_WORKERS

if [ $? -ne 0 ]; then
    echo "Preprocessing failed!"
    exit 1
fi

# 第二步：启动worker处理
echo "Step 2: Starting workers..."
python monitor.py --total-workers $TOTAL_WORKERS \
                           --cache-dir $CACHE_DIR \
                           --output-dir $OUTPUT_DIR \
                           --status-dir $STATUS_DIR \
                           --log-dir $LOG_DIR

# 第三步：合并结果
if [ $? -eq 0 ]; then
    echo "Step 3: Merging results..."
    python merge_results.py --output-dir $OUTPUT_DIR \
                           --total-workers $TOTAL_WORKERS \
                           --final-output $FINAL_OUTPUT
    echo "Processing complete! Final output: $FINAL_OUTPUT"
    
    # 可选：清理缓存文件
    echo "Cleaning up cache files..."
    rm -rf $CACHE_DIR
else
    echo "Processing was interrupted or failed."
    echo "Cache files preserved at: $CACHE_DIR for debugging"
fi
