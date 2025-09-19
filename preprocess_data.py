import os
import numpy as np
import h5py
from pathlib import Path
import argparse
from tqdm import tqdm
import shutil

def preprocess_data(original_cache_root, output_dir, total_workers):
    """预处理数据：将数据集分割成worker专用的缓存文件"""
    print("Starting data preprocessing...")
    
    # 设置原始缓存路径
    os.environ['SEISBENCH_CACHE_ROOT'] = original_cache_root
    import seisbench.data as sbd
    
    # 加载数据集
    dataset = sbd.STEAD()
    total_samples = len(dataset)
    print(f"Total samples: {total_samples}")
    
    # 计算每个worker处理的样本范围
    chunk_size = total_samples // total_workers
    ranges = []
    for i in range(total_workers):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < total_workers - 1 else total_samples
        ranges.append((start, end))
    
    # 为每个worker创建缓存文件
    for worker_id, (start, end) in enumerate(tqdm(ranges, desc="Creating worker caches")):
        worker_samples = end - start
        cache_file = Path(output_dir) / f'worker_{worker_id}_cache.h5'
        
        with h5py.File(cache_file, 'w') as f:
            # 创建数据集
            waveforms_dset = f.create_dataset(
                'waveforms', 
                shape=(worker_samples, 3, 6000), 
                dtype=np.float32
            )
            
            # 批量读取和保存数据
            batch_size = 100
            for batch_start in range(0, worker_samples, batch_size):
                batch_end = min(batch_start + batch_size, worker_samples)
                
                # 读取原始数据
                batch_data = []
                for idx in range(start + batch_start, start + batch_end):
                    sample = dataset.get_waveforms(idx)
                    batch_data.append(sample)
                
                # 保存到缓存文件
                waveforms_dset[batch_start:batch_end] = np.array(batch_data)
        
        print(f"Worker {worker_id}: Created cache with {worker_samples} samples")
    
    # 保存元数据
    metadata = {
        'total_samples': total_samples,
        'total_workers': total_workers,
        'worker_ranges': ranges
    }
    
    np.save(Path(output_dir) / 'metadata.npy', metadata)
    print("Preprocessing completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess STEAD data for parallel processing')
    parser.add_argument('--original-cache-root', type=str, required=True, help='Original seisbench cache root')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for worker caches')
    parser.add_argument('--total-workers', type=int, default=32, help='Total number of workers')
    
    args = parser.parse_args()
    
    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    preprocess_data(args.original_cache_root, args.output_dir, args.total_workers)
