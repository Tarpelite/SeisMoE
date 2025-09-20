#!/bin/bash

# 设置变量
TOTAL_WORKERS=16
ORIGINAL_CACHE_ROOT="/home/icassp2026/tianyu/local_cache"
CACHE_DIR="/home/icassp2026/tianyu/local_cache/datasets/stead/worker_caches"  # worker缓存文件目录
OUTPUT_DIR="./stead/output"
STATUS_DIR="./stead/status"
LOG_DIR="./stead/logs"
FINAL_OUTPUT="STEAD_emd_final.hdf5"

# 创建目录
mkdir -p $CACHE_DIR $OUTPUT_DIR $STATUS_DIR $LOG_DIR

echo "Starting optimized stead EMD processing..."
echo "Total workers: $TOTAL_WORKERS"

# # # 第一步：预处理数据
# echo "Step 1: Preprocessing data..."
# python /home/icassp2026/emd_extract/SeisMoE/preprocess_data_stead.py  \
#                             --input-file $ORIGINAL_CACHE_ROOT/datasets/stead/waveforms.hdf5 \
#                             --output-dir $CACHE_DIR

# if [ $? -ne 0 ]; then
#     echo "Preprocessing failed!"
#     exit 1
# fi

# # 第二步：启动worker处理
echo "Step 2: Starting workers..."
python monitor_stead.py --total-workers $TOTAL_WORKERS \
                           --cache-dir $CACHE_DIR \
                           --output-dir $OUTPUT_DIR \
                           --status-dir $STATUS_DIR \
                           --log-dir $LOG_DIR

# # 第三步：合并结果
# if [ $? -eq 0 ]; then
#     echo "Step 3: Merging results..."
#     python /home/icassp2026/emd_extract/SeisMoE/merge_geofon.py --output-dir $OUTPUT_DIR \
#                            --total-workers $TOTAL_WORKERS \
#                            --final-output $FINAL_OUTPUT
#     echo "Processing complete! Final output: $FINAL_OUTPUT"
    
#     # 可选：清理缓存文件
#     echo "Cleaning up cache files..."
#     rm -rf $CACHE_DIR
# else
#     echo "Processing was interrupted or failed."
#     echo "Cache files preserved at: $CACHE_DIR for debugging"
# fi
