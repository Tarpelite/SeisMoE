#!/usr/bin/env python3
"""
从local_cache中读取waveforms.hdf5文件，随机绘制一个样本的原始波形图
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
import argparse

def plot_random_waveform(file_path, output_dir=None):
    """随机选择并绘制一个样本的波形图"""
    
    if output_dir is None:
        output_dir = Path(file_path).parent
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    
    with h5py.File(file_path, 'r') as f:
        data_group = f['data']
        dataset_names = list(data_group.keys())
        
        # 随机选择一个数据集
        #chosen_dataset = random.choice(dataset_names)
        chosen_dataset = 'bucket10'
        print(f"选择的数据集: {chosen_dataset}")
        
        # 读取数据集
        dataset = data_group[chosen_dataset]
        data = dataset[:]
        print(f"数据形状: {data.shape}")
        
        # 处理不同的数据形状，随机选择一个样本
        if len(data.shape) == 3:
            # 3维数据 (samples, channels, length)
            num_samples = data.shape[0]
            sample_idx = random.randint(0, num_samples - 1)
            sample_data = data[sample_idx]
        elif len(data.shape) == 2:
            # 2维数据，假设为单个样本
            sample_data = data
            sample_idx = 0
        else:
            print(f"不支持的数据形状: {data.shape}")
            return None
        
        # 确保数据为 (channels, length) 格式
        if len(sample_data.shape) == 2:
            if sample_data.shape[1] == 3:
                sample_data = sample_data.T  # 转置
        
        # 确保有3个通道
        if sample_data.shape[0] < 3:
            padding = np.zeros((3 - sample_data.shape[0], sample_data.shape[1]))
            sample_data = np.vstack([sample_data, padding])
        elif sample_data.shape[0] > 3:
            sample_data = sample_data[:3, :]
        
        # 绘制波形图
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        
        channel_names = ['Channel 1 (Z)', 'Channel 2 (N)', 'Channel 3 (E)']
        colors = ['red', 'blue', 'green']
        
        time_axis = np.arange(sample_data.shape[1])
        
        for i in range(3):
            axes[i].plot(time_axis, sample_data[i], color=colors[i], linewidth=1.0)
            axes[i].set_title(f'{channel_names[i]} - {chosen_dataset} Sample {sample_idx}', 
                            fontweight='bold', fontsize=12)
            axes[i].set_ylabel('Amplitude')
            axes[i].grid(True, alpha=0.3)
        
        axes[2].set_xlabel('Time Samples')
        
        # 添加整体标题
        dataset_path = Path(file_path).parent.name
        fig.suptitle(f'Original Waveform - {dataset_path.upper()} Dataset\n'
                    f'Dataset: {chosen_dataset}, Sample: {sample_idx}', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # 保存图片
        save_filename = f"{dataset_path}_{chosen_dataset}_sample{sample_idx}_waveform.png"
        save_path = output_dir / save_filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"波形图已保存: {save_path}")
        return str(save_path)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='从local_cache中随机绘制波形图')
    parser.add_argument('--file', type=str, 
                       default='/home/icassp2026/tianyu/local_cache/datasets/stead/waveforms.hdf5',
                       help='waveforms.hdf5文件路径')
    parser.add_argument('--output-dir', type=str,
                       help='输出目录（默认为文件所在目录）')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not Path(args.file).exists():
        print(f"文件不存在: {args.file}")
        return
    
    # 绘制随机波形
    try:
        save_path = plot_random_waveform(args.file, args.output_dir)
        if save_path:
            print("波形绘制完成!")
        
    except Exception as e:
        print(f"绘制波形时发生错误: {e}")

if __name__ == "__main__":
    main()
