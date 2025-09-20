import os
import numpy as np
import h5py
from pathlib import Path
from PyEMD import EMD
import time
import argparse
import signal
import json
from datetime import datetime
import logging

# 配置日志
def setup_logging(worker_id):
    """设置日志配置"""
    log_dir = Path("./stead/logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f'worker_{worker_id}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - Worker %(worker_id)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    
    # 添加worker_id到日志格式
    old_factory = logging.getLogRecordFactory()
    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.worker_id = worker_id
        return record
    logging.setLogRecordFactory(record_factory)

class GracefulKiller:
    """优雅退出处理"""
    def __init__(self, worker_id):
        self.kill_now = False
        self.worker_id = worker_id
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
    
    def exit_gracefully(self, signum, frame):
        logging.info("Received termination signal")
        self.kill_now = True

class WorkerDataLoader:
    """高效的数据加载器，支持多 dataset 结构（懒加载模式）"""
    def __init__(self, cache_file, worker_id):
        self.cache_file = cache_file
        self.worker_id = worker_id
        self.datasets = []   # 保存 dataset 名称
        self.dataset_info = {}  # 保存 dataset 的元信息
        self.total_samples = 0
        self.analyze_structure()
    
    def analyze_structure(self):
        """分析数据结构，不加载实际数据"""
        logging.info(f"Analyzing data structure from {self.cache_file}...")
        start_time = time.time()
        
        try:
            with h5py.File(self.cache_file, 'r') as f:
                # 你的缓存文件结构是 data/xxx
                if "data" not in f:
                    raise ValueError("cache file missing 'data' group")
                
                grp = f["data"]
                self.datasets = list(grp.keys())  # 所有 dataset 名称
                logging.info(f"Found {len(self.datasets)} datasets in worker cache")
                
                # 只分析结构，不加载数据
                for name in self.datasets:
                    dataset = grp[name]
                    self.dataset_info[name] = {
                        'shape': dataset.shape,
                        'dtype': dataset.dtype,
                        'samples': dataset.shape[0]
                    }
                    self.total_samples += dataset.shape[0]
                    logging.info(f"Dataset {name}: shape={dataset.shape}, dtype={dataset.dtype}")
            
            analyze_time = time.time() - start_time
            logging.info(
                f"Structure analyzed in {analyze_time:.2f}s, "
                f"datasets: {len(self.datasets)}, total samples: {self.total_samples}"
            )
            
        except Exception as e:
            logging.error(f"Failed to analyze data structure from {self.cache_file}: {e}")
            raise
    
    def load_dataset(self, dataset_name):
        """懒加载指定的dataset"""
        try:
            with h5py.File(self.cache_file, 'r') as f:
                return f["data"][dataset_name][:]
        except Exception as e:
            logging.error(f"Failed to load dataset {dataset_name}: {e}")
            raise
    
    def __len__(self):
        """总样本数"""
        return self.total_samples
    
    def get_dataset_count(self):
        """获取dataset数量"""
        return len(self.datasets)
    
    def get_dataset_info(self, dataset_name):
        """获取dataset信息"""
        return self.dataset_info.get(dataset_name)

def ensure_3channel_format(waveform):
    """确保波形数据为(3, length)格式"""
    waveform = np.array(waveform)
    
    # 如果是(length, 3)格式，转置为(3, length)
    if len(waveform.shape) == 2 and waveform.shape[1] == 3:
        return waveform.T
    
    # 如果已经是(3, length)格式，直接返回
    if len(waveform.shape) == 2 and waveform.shape[0] == 3:
        return waveform
    
    # 其他情况记录警告但仍然处理
    logging.warning(f"Unexpected waveform shape: {waveform.shape}, assuming first dimension is channels")
    return waveform

def format_time(seconds):
    """格式化时间显示"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"

def process_worker(worker_id, cache_file, output_dir, status_file):
    """处理单个工作进程"""
    # 设置日志
    setup_logging(worker_id)
    killer = GracefulKiller(worker_id)
    
    # 初始化状态
    status = {
        'worker_id': worker_id,
        'processed': 0,
        'total_samples': 0,
        'start_time': datetime.now().isoformat(),
        'last_update': datetime.now().isoformat(),
        'status': 'running',
        'error': None
    }
    
    def update_status(processed=None, status_msg=None, error=None):
        if processed is not None:
            status['processed'] = processed
        if status_msg:
            status['status'] = status_msg
        if error:
            status['error'] = str(error)
        status['last_update'] = datetime.now().isoformat()
        
        try:
            with open(status_file, 'w') as f:
                json.dump(status, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to update status file: {e}")
    
    try:
        logging.info(f"Starting worker {worker_id} with cache file: {cache_file}")
        
        # 加载数据到内存
        data_loader = WorkerDataLoader(cache_file, worker_id)
        total_samples = len(data_loader)
        status['total_samples'] = total_samples
        update_status()
        
        logging.info(f"Loaded {total_samples} samples from {data_loader.get_dataset_count()} datasets into memory")
        
        # 预初始化EMD实例
        emd_instances = [EMD(max_imfs=3, max_iterations=500) for _ in range(3)]
        
        # 输出文件路径
        output_path = Path(output_dir) / f'STEAD_emd_worker_{worker_id}.hdf5'
        
        # 处理数据 - 按dataset顺序处理
        with h5py.File(output_path, 'w') as hdf5_file:
            # 创建一个组来存储所有dataset的IMF结果
            imf_group = hdf5_file.create_group('IMFs')
            
            processed_count = 0
            start_time = time.time()
            last_log_time = start_time
            
            # 按原始dataset顺序处理
            for dataset_idx, dataset_name in enumerate(data_loader.datasets):
                if killer.kill_now:
                    logging.warning("Received termination signal, stopping processing")
                    update_status(status_msg='terminated')
                    break
                
                try:
                    # 懒加载当前dataset的数据
                    dataset_data = data_loader.load_dataset(dataset_name)  # shape: (samples_in_dataset, ...)
                    dataset_samples = dataset_data.shape[0]
                    
                    logging.info(f"Processing dataset {dataset_name}: {dataset_samples} samples, shape: {dataset_data.shape}")
                    
                    # 处理这个dataset中的所有样本
                    dataset_imfs = []
                    for sample_idx in range(dataset_samples):
                        try:
                            # 获取单个样本
                            raw_sample = dataset_data[sample_idx]
                            
                            # 确保为(3, length)格式
                            sample = ensure_3channel_format(raw_sample)
                            sample_length = sample.shape[1]
                            
                            # EMD处理
                            sample_IMFs = []
                            for channel_idx in range(3):
                                IMFs = emd_instances[channel_idx](sample[channel_idx])
                                first_3 = IMFs[:3,:] if IMFs.shape[0] >= 3 else IMFs
                                if IMFs.shape[0] < 3:
                                    padding = np.zeros((3 - IMFs.shape[0], sample_length))
                                    first_3 = np.vstack((IMFs, padding))
                                sample_IMFs.append(first_3)
                            
                            dataset_imfs.append(np.array(sample_IMFs))  # shape: (3, 3, length)
                            processed_count += 1
                            
                        except Exception as e:
                            logging.error(f"Error processing sample {sample_idx} in dataset {dataset_name}: {e}")
                            # 使用零填充，长度与dataset中第一个样本相同
                            if len(dataset_imfs) > 0:
                                ref_length = dataset_imfs[0].shape[2]
                            else:
                                ref_length = sample.shape[1] if 'sample' in locals() else 1000
                            zero_result = np.zeros((3, 3, ref_length), dtype=np.float32)
                            dataset_imfs.append(zero_result)
                            processed_count += 1
                    
                    # 将整个dataset的IMF结果保存为一个dataset
                    if dataset_imfs:
                        dataset_imf_array = np.array(dataset_imfs)  # shape: (samples, 3, 3, length)
                        imf_group.create_dataset(dataset_name, data=dataset_imf_array, dtype=np.float32)
                        logging.info(f"Saved dataset {dataset_name} IMFs with shape: {dataset_imf_array.shape}")
                    
                except Exception as e:
                    logging.error(f"Error processing dataset {dataset_name}: {e}")
                    continue
                
                # 更新状态和记录进度
                current_time = time.time()
                elapsed_time = current_time - start_time
                
                # 每处理100个样本或每30秒记录一次进度
                if (processed_count % 100 == 0 or 
                    current_time - last_log_time > 30 or 
                    processed_count == total_samples):
                    
                    # 计算预估剩余时间
                    if processed_count > 0:
                        samples_per_second = processed_count / elapsed_time
                        remaining_samples = total_samples - processed_count
                        if samples_per_second > 0:
                            eta_seconds = remaining_samples / samples_per_second
                            eta_str = format_time(eta_seconds)
                        else:
                            eta_str = "calculating..."
                        
                        progress_percent = (processed_count / total_samples) * 100
                        logging.info(
                            f"Progress: {processed_count}/{total_samples} "
                            f"({progress_percent:.1f}%), "
                            f"ETA: {eta_str}, "
                            f"Speed: {samples_per_second:.2f} samples/s"
                        )
                    
                    update_status(processed=processed_count)
                    last_log_time = current_time
            
            # 最终完成状态
            total_time = time.time() - start_time
            avg_speed = total_samples / total_time if total_time > 0 else 0
            logging.info(
                f"Completed processing {processed_count} samples in {format_time(total_time)}, "
                f"average speed: {avg_speed:.2f} samples/s"
            )
            update_status(processed=processed_count, status_msg='completed')
    
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
        update_status(status_msg='failed', error=str(e))
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimized STEAD EMD Processing Worker')
    parser.add_argument('--worker-id', type=int, required=True, help='Worker ID')
    parser.add_argument('--cache-dir', type=str, required=True, help='Directory with worker cache files')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--status-dir', type=str, required=True, help='Status directory')
    
    args = parser.parse_args()
    
    # 确保目录存在
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.status_dir).mkdir(parents=True, exist_ok=True)
    
    # 缓存文件路径
    cache_file = Path(args.cache_dir) / f'worker_{args.worker_id}_cache.h5'
    status_file = Path(args.status_dir) / f'worker_{args.worker_id}.json'
    
    if not cache_file.exists():
        logging.error(f"Cache file not found: {cache_file}")
        exit(1)
    
    process_worker(args.worker_id, str(cache_file), args.output_dir, status_file)
