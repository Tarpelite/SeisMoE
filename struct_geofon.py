import h5py
import numpy as np

def explore_h5(name, obj):
    """递归打印 HDF5 文件结构"""
    if isinstance(obj, h5py.Group):
        print(f"[Group] {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"[Dataset] {name}, shape={obj.shape}, dtype={obj.dtype}, size={obj.size}")
    
    # 打印属性
    for key, val in obj.attrs.items():
        print(f"    [Attribute] {name} -> {key}: {val}")

def list_h5_structure(file_path):
    with h5py.File(file_path, "r") as f:
        print(f"File: {file_path}")
        f.visititems(explore_h5)

# 使用示例
list_h5_structure("/home/icassp2026/tianyu/local_cache/datasets/instancecounts/waveforms.hdf5")
#list_h5_structure("/home/icassp2026/emd_extract/SeisMoE/output/stead_emd_worker_15.hdf5")
#list_h5_structure("/home/icassp2026/tianyu/local_cache/datasets/stead/worker_caches/worker_0_cache.h5")