# check_imf.py - 单样本原始波形与IMF分解对比
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

# 文件路径
imf_file_path = Path("/home/icassp2026/emd_extract/SeisMoE/ethz/output/ETHZ_emd_worker_13.hdf5")
cache_file_path = Path("/home/icassp2026/tianyu/local_cache/datasets/ethz/worker_caches/worker_13_cache.h5")

# 输出目录（当前目录）
output_dir = Path(".")
output_dir.mkdir(exist_ok=True)

def plot_single_sample_analysis():
    """随机选择一个样本，绘制原始波形和IMF分解结果"""
    
    print("Reading files...")
    
    # 打开IMF文件，获取所有buckets
    with h5py.File(imf_file_path, 'r') as imf_f:
        imf_group = imf_f['IMFs']
        buckets = list(imf_group.keys())
        print(f"Found {len(buckets)} buckets in IMF file")
        
        if not buckets:
            print("No buckets found!")
            return
        
        # 随机选择一个bucket
        chosen_bucket = random.choice(buckets)
        #chosen_bucket = "bucket32"
        print(f"Chosen bucket: {chosen_bucket}")
        
        # 读取IMF数据
        imf_data = imf_group[chosen_bucket][:]  # shape: (samples, 3, 3, length)
        print(f"IMF data shape: {imf_data.shape}")
        
        # 随机选择一个样本
        num_samples = imf_data.shape[0]
        chosen_sample_idx = random.randint(0, num_samples - 1)
        print(f"Chosen sample index: {chosen_sample_idx} (out of {num_samples})")
        
        # 提取选中样本的IMF数据
        sample_imf = imf_data[chosen_sample_idx]  # shape: (3, 3, length)
        
    # 从cache文件中读取对应的原始数据
    print("Reading original data from cache...")
    with h5py.File(cache_file_path, 'r') as cache_f:
        data_group = cache_f['data']
        
        if chosen_bucket not in data_group:
            print(f"Bucket {chosen_bucket} not found in cache file!")
            return
        
        # 读取原始数据
        original_data = data_group[chosen_bucket][:]
        print(f"Original data shape: {original_data.shape}")
        
        # 提取选中样本的原始数据
        if len(original_data.shape) == 3:  # (samples, 3, length)
            sample_original = original_data[chosen_sample_idx]  # shape: (3, length)
        elif len(original_data.shape) == 2:  # (3, length) - 只有一个样本
            if chosen_sample_idx > 0:
                print("Warning: Only one sample available, using index 0")
            sample_original = original_data  # shape: (3, length)
        else:
            print(f"Unexpected data shape: {original_data.shape}")
            return
    
    # 创建对比图
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    
    # 第一个子图：原始波形（第一个通道）
    axes[0].plot(sample_original[0], 'black', linewidth=1.5, label='Original Waveform')
    axes[0].set_title(f'{chosen_bucket} - Sample {chosen_sample_idx} - Original Waveform (Channel 1)', 
                     fontweight='bold', fontsize=12)
    axes[0].set_ylabel('Amplitude')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 第二、三、四个子图：三个IMF分量（第一个通道）
    imf_colors = ['red', 'blue', 'green']
    imf_names = ['IMF1 (High Frequency)', 'IMF2 (Medium Frequency)', 'IMF3 (Low Frequency)']
    
    for i in range(3):
        # 提取第一个通道的第i个IMF
        imf_component = sample_imf[0, i, :]  # shape: (length,)
        
        axes[i+1].plot(imf_component, color=imf_colors[i], linewidth=1.5, label=imf_names[i])
        axes[i+1].set_title(f'{chosen_bucket} - Sample {chosen_sample_idx} - {imf_names[i]} (Channel 1)', 
                           fontweight='bold', fontsize=12)
        axes[i+1].set_ylabel('Amplitude')
        axes[i+1].legend()
        axes[i+1].grid(True, alpha=0.3)
    
    axes[3].set_xlabel('Time Samples')
    
    plt.tight_layout()
    
    # 保存图片
    save_path = output_dir / f"ethz_{chosen_bucket}_sample{chosen_sample_idx}_waveform_imf_analysis.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved analysis: {save_path}")
    
    # 打印统计信息
    print("\n" + "="*60)
    print("Sample Analysis Summary")
    print("="*60)
    print(f"Bucket: {chosen_bucket}")
    print(f"Sample Index: {chosen_sample_idx}")
    print(f"Original waveform length: {len(sample_original[0])}")
    print(f"Original waveform stats (Channel 1):")
    print(f"  Mean: {np.mean(sample_original[0]):.4f}")
    print(f"  Std:  {np.std(sample_original[0]):.4f}")
    print(f"  Min:  {np.min(sample_original[0]):.4f}")
    print(f"  Max:  {np.max(sample_original[0]):.4f}")
    
    print(f"\nIMF Components (Channel 1):")
    for i in range(3):
        imf_comp = sample_imf[0, i, :]
        print(f"  {imf_names[i]}:")
        print(f"    Mean: {np.mean(imf_comp):.4f}")
        print(f"    Std:  {np.std(imf_comp):.4f}")
        print(f"    Min:  {np.min(imf_comp):.4f}")
        print(f"    Max:  {np.max(imf_comp):.4f}")
    
    return chosen_bucket, chosen_sample_idx

if __name__ == "__main__":
    # 检查文件是否存在
    if not imf_file_path.exists():
        print(f"IMF file not found: {imf_file_path}")
        exit(1)
    
    if not cache_file_path.exists():
        print(f"Cache file not found: {cache_file_path}")
        exit(1)
    
    # 执行分析
    result = plot_single_sample_analysis()
    
    if result:
        bucket, sample_idx = result
        print(f"\nAnalysis completed for {bucket}, sample {sample_idx}!")
    else:
        print("Analysis failed!")
        exit(1)
