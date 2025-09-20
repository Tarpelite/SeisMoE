import h5py
from pathlib import Path
import argparse
from tqdm import tqdm
import time

def preprocess_data(input_file, output_dir):
    """将HDF5文件拆分为16个文件"""
    print("开始数据预处理...")
    start_time = time.time()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    num_splits = 16  # 拆成 16 个文件
    
    with h5py.File(input_file, "r") as f_in:
        datasets = list(f_in["data"].keys())  # 所有 dataset 名字，保持原顺序
        total = len(datasets)
        print(f"总数据集数量: {total}")

        # 每份大小
        split_size = total // num_splits
        remainder = total % num_splits

        start = 0
        cache_times = []
        
        for part in tqdm(range(num_splits), desc="创建分割文件"):
            cache_start_time = time.time()
            # 每份分配 split_size 个，如果有余数前几份多分一个
            end = start + split_size + (1 if part < remainder else 0)
            part_datasets = datasets[start:end]

            out_file = output_dir / f"worker_{part}_cache.h5"
            with h5py.File(out_file, "w") as f_out:
                grp_out = f_out.create_group("data")
                bucket_times = []
                
                for name in part_datasets:
                    bucket_start_time = time.time()
                    f_in.copy(f"data/{name}", grp_out)  # 原模原样复制 dataset
                    bucket_time = time.time() - bucket_start_time
                    bucket_times.append(bucket_time)
                    print(f"复制 {name} 完成 (耗时 {bucket_time:.3f}s)")
                
                # 统计单个bucket平均时间
                avg_bucket_time = sum(bucket_times) / len(bucket_times) if bucket_times else 0

            cache_time = time.time() - cache_start_time
            cache_times.append(cache_time)
            
            # 预估剩余时间
            if len(cache_times) > 0:
                avg_cache_time = sum(cache_times) / len(cache_times)
                remaining_caches = num_splits - (part + 1)
                estimated_remaining_time = remaining_caches * avg_cache_time
                
                print(f"已写入 {out_file}, 数据集数量: {len(part_datasets)} 个")
                print(f"缓存文件处理时间: {cache_time:.2f}s")
                print(f"平均单个bucket处理时间: {avg_bucket_time:.3f}s")
                print(f"预估每个缓存文件处理时间: {avg_cache_time:.2f}s")
                print(f"预估剩余时间: {estimated_remaining_time:.2f}s ({estimated_remaining_time/60:.1f} 分钟)")
                print("-" * 50)
            start = end
    
    total_time = time.time() - start_time
    print(f"预处理完成! 总耗时: {total_time:.2f}s ({total_time/60:.1f} 分钟)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='将HDF5文件拆分为16个部分')
    parser.add_argument('--input-file', type=str, required=True, help='输入HDF5文件路径')
    parser.add_argument('--output-dir', type=str, required=True, help='分割文件输出目录')
    
    args = parser.parse_args()
    
    preprocess_data(args.input_file, args.output_dir)
