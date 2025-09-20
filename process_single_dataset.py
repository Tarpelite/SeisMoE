#!/usr/bin/env python3
"""
处理单个worker中有问题的数据集
专门处理形状异常的数据集，并将结果追加到现有的EMD输出文件中
"""
import os
import numpy as np
import h5py
from pathlib import Path
from PyEMD import EMD
import time
import logging
from datetime import datetime

def setup_logging():
    """设置日志配置"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / 'process_single_dataset.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def fix_waveform_shape(waveform):
    """修复波形数据的形状问题"""
    waveform = np.array(waveform)
    
    logging.info(f"Original waveform shape: {waveform.shape}")
    
    # 处理不同的形状情况
    if len(waveform.shape) == 1:
        # 1维数据 (length,) -> 复制为3个通道
        length = waveform.shape[0]
        fixed_waveform = np.tile(waveform, (3, 1))  # shape: (3, length)
        logging.info(f"Fixed 1D data: {waveform.shape} -> {fixed_waveform.shape}")
        
    elif len(waveform.shape) == 2:
        if waveform.shape[0] == 3:
            # 已经是 (3, length) 格式
            fixed_waveform = waveform
            logging.info("Data already in correct (3, length) format")
        elif waveform.shape[1] == 3:
            # (length, 3) -> 转置为 (3, length)
            fixed_waveform = waveform.T
            logging.info(f"Transposed data: {waveform.shape} -> {fixed_waveform.shape}")
        else:
            # 其他2D情况，假设第一维是通道
            if waveform.shape[0] <= 3:
                # 补充到3个通道
                channels_needed = 3 - waveform.shape[0]
                if channels_needed > 0:
                    padding = np.zeros((channels_needed, waveform.shape[1]))
                    fixed_waveform = np.vstack([waveform, padding])
                else:
                    fixed_waveform = waveform[:3, :]
                logging.info(f"Padded/trimmed channels: {waveform.shape} -> {fixed_waveform.shape}")
            else:
                # 取前3个通道
                fixed_waveform = waveform[:3, :]
                logging.info(f"Trimmed to 3 channels: {waveform.shape} -> {fixed_waveform.shape}")
    else:
        # 多维数据，尝试reshape
        logging.warning(f"Unexpected shape {waveform.shape}, attempting to flatten and reshape")
        flat_data = waveform.flatten()
        # 尝试分成3个通道
        length = len(flat_data) // 3
        if length > 0:
            fixed_waveform = flat_data[:length*3].reshape(3, length)
        else:
            # 如果数据太少，就复制
            fixed_waveform = np.tile(flat_data, (3, 1))
        logging.info(f"Reshaped data: {waveform.shape} -> {fixed_waveform.shape}")
    
    return fixed_waveform

def process_problematic_datasets():
    """处理有问题的数据集"""
    logging.info("开始处理有问题的数据集...")
    
    # 文件路径
    cache_file = "/mnt/samba/seisbench_cache/datasets/geofon/worker_caches/worker_15_cache.h5"
    output_file = "/home/icassp2026/emd_extract/SeisMoE/output/GEOFON_emd_worker_15.hdf5"
    
    # 需要处理的问题数据集（从日志中识别的）
    problematic_datasets = [
        "gfz2011yrrp_IU.OTAV.00.BH",
        "gfz2012mnsm_NO.AKN.00.BH", 
        "gfz2012sasl_IU.MA2.00.BH",
        "gfz2013cnwn_GE.TNTI..BH1"
    ]
    
    # 检查文件是否存在
    if not Path(cache_file).exists():
        logging.error(f"Cache file not found: {cache_file}")
        return False
    
    if not Path(output_file).exists():
        logging.error(f"Output file not found: {output_file}")
        return False
    
    # 预初始化EMD实例
    emd_instances = [EMD(max_imfs=3, max_iterations=500) for _ in range(3)]
    logging.info("EMD instances initialized")
    
    processed_count = 0
    start_time = time.time()
    
    # 读取cache文件
    with h5py.File(cache_file, 'r') as cache_f:
        data_group = cache_f['data']
        available_datasets = list(data_group.keys())
        logging.info(f"Available datasets in cache: {len(available_datasets)}")
        
        # 找到实际存在的问题数据集
        existing_problematic = [ds for ds in problematic_datasets if ds in available_datasets]
        logging.info(f"Found {len(existing_problematic)} problematic datasets to process")
        
        if not existing_problematic:
            logging.warning("No problematic datasets found in cache file")
            return True
        
        # 处理每个问题数据集
        results_to_append = {}
        
        for dataset_name in existing_problematic:
            logging.info(f"Processing dataset: {dataset_name}")
            
            try:
                # 读取原始数据
                original_data = data_group[dataset_name][:]
                logging.info(f"Original data shape: {original_data.shape}")
                
                # 确定样本数
                if len(original_data.shape) == 3:  # (samples, channels, length)
                    num_samples = original_data.shape[0]
                elif len(original_data.shape) == 2:  # (channels, length) - 单个样本
                    num_samples = 1
                    original_data = original_data[np.newaxis, ...]  # 添加样本维度
                else:
                    logging.error(f"Cannot determine sample count for shape: {original_data.shape}")
                    continue
                
                logging.info(f"Processing {num_samples} samples")
                
                # 处理每个样本
                dataset_imfs = []
                for sample_idx in range(num_samples):
                    try:
                        # 获取单个样本
                        if len(original_data.shape) == 3:
                            raw_sample = original_data[sample_idx]
                        else:
                            raw_sample = original_data
                        
                        # 修复形状问题
                        fixed_sample = fix_waveform_shape(raw_sample)  # shape: (3, length)
                        sample_length = fixed_sample.shape[1]
                        
                        logging.info(f"Sample {sample_idx}: fixed shape = {fixed_sample.shape}")
                        
                        # EMD处理
                        sample_IMFs = []
                        for channel_idx in range(3):
                            try:
                                channel_data = fixed_sample[channel_idx]
                                IMFs = emd_instances[channel_idx](channel_data)
                                
                                # 取前3个IMF，不足则补零
                                first_3 = IMFs[:3, :] if IMFs.shape[0] >= 3 else IMFs
                                if IMFs.shape[0] < 3:
                                    padding = np.zeros((3 - IMFs.shape[0], sample_length))
                                    first_3 = np.vstack((IMFs, padding))
                                
                                sample_IMFs.append(first_3)
                                
                            except Exception as e:
                                logging.error(f"EMD failed for channel {channel_idx}: {e}")
                                # 使用零填充
                                zero_imfs = np.zeros((3, sample_length))
                                sample_IMFs.append(zero_imfs)
                        
                        # 组合结果 - shape: (3, 3, length)
                        result_array = np.array(sample_IMFs)
                        dataset_imfs.append(result_array)
                        processed_count += 1
                        
                        logging.info(f"Successfully processed sample {sample_idx}, result shape: {result_array.shape}")
                        
                    except Exception as e:
                        logging.error(f"Error processing sample {sample_idx} in dataset {dataset_name}: {e}")
                        # 使用默认大小的零填充
                        default_length = 1000
                        zero_result = np.zeros((3, 3, default_length))
                        dataset_imfs.append(zero_result)
                        processed_count += 1
                        continue
                
                # 将整个数据集的结果保存
                if dataset_imfs:
                    dataset_imf_array = np.array(dataset_imfs)  # shape: (samples, 3, 3, length)
                    results_to_append[dataset_name] = dataset_imf_array
                    logging.info(f"Dataset {dataset_name} processed successfully, final shape: {dataset_imf_array.shape}")
                
            except Exception as e:
                logging.error(f"Error processing dataset {dataset_name}: {e}")
                continue
    
    # 将结果追加到输出文件
    if results_to_append:
        logging.info("Appending results to output file...")
        
        with h5py.File(output_file, 'a') as output_f:  # 'a' 模式用于追加
            imf_group = output_f['IMFs']
            
            for dataset_name, data in results_to_append.items():
                try:
                    # 检查是否已存在
                    if dataset_name in imf_group:
                        logging.warning(f"Dataset {dataset_name} already exists, skipping...")
                        continue
                    
                    # 创建新的数据集
                    imf_group.create_dataset(dataset_name, data=data, dtype=np.float32)
                    logging.info(f"Added dataset {dataset_name} to output file, shape: {data.shape}")
                    
                except Exception as e:
                    logging.error(f"Error saving dataset {dataset_name}: {e}")
                    continue
    
    total_time = time.time() - start_time
    avg_speed = processed_count / total_time if total_time > 0 else 0
    
    logging.info("=" * 60)
    logging.info("处理完成!")
    logging.info(f"处理的样本数: {processed_count}")
    logging.info(f"处理的数据集数: {len(results_to_append)}")
    logging.info(f"总耗时: {total_time:.2f}s")
    logging.info(f"平均速度: {avg_speed:.2f} samples/s")
    logging.info("=" * 60)
    
    return True

if __name__ == "__main__":
    # 设置日志
    setup_logging()
    
    try:
        success = process_problematic_datasets()
        if success:
            logging.info("所有问题数据集处理完成!")
        else:
            logging.error("处理过程中出现错误")
            exit(1)
            
    except Exception as e:
        logging.error(f"程序执行失败: {e}", exc_info=True)
        exit(1)
