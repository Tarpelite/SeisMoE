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
    log_dir = Path("logs")
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
    """高效的数据加载器，直接从缓存文件读取"""
    def __init__(self, cache_file, worker_id):
        self.cache_file = cache_file
        self.data = None
        self.worker_id = worker_id
        self.load_data()
    
    def load_data(self):
        """将整个worker的数据加载到内存"""
        logging.info(f"Loading data from {self.cache_file}...")
        start_time = time.time()
        
        try:
            with h5py.File(self.cache_file, 'r') as f:
                # 一次性加载所有数据到内存
                self.data = f['waveforms'][:]
            
            load_time = time.time() - start_time
            logging.info(f"Data loaded in {load_time:.2f}s, shape: {self.data.shape}")
            
        except Exception as e:
            logging.error(f"Failed to load data from {self.cache_file}: {e}")
            raise
    
    def get_sample(self, index):
        """获取单个样本"""
        return self.data[index]

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
        total_samples = len(data_loader.data)
        status['total_samples'] = total_samples
        update_status()
        
        logging.info(f"Loaded {total_samples} samples into memory")
        
        # 预初始化EMD实例
        emd_instances = [EMD(max_imfs=3, max_iterations=500) for _ in range(3)]
        
        # 输出文件路径
        output_path = Path(output_dir) / f'STEAD_emd_worker_{worker_id}.hdf5'
        
        # 处理数据
        with h5py.File(output_path, 'w') as hdf5_file:
            dset = hdf5_file.create_dataset(
                'IMFs', 
                shape=(total_samples, 3, 3, 6000), 
                dtype=np.float32
            )
            
            processed_count = 0
            batch_size = 32
            start_time = time.time()
            last_log_time = start_time
            
            for batch_start in range(0, total_samples, batch_size):
                if killer.kill_now:
                    logging.warning("Received termination signal, stopping processing")
                    update_status(status_msg='terminated')
                    break
                
                batch_end = min(batch_start + batch_size, total_samples)
                batch_results = []
                
                # 处理批次
                for index in range(batch_start, batch_end):
                    try:
                        # 从内存中获取数据（非常快）
                        sample = data_loader.get_sample(index)
                        
                        # EMD处理
                        sample_IMFs = []
                        for i in range(3):
                            IMFs = emd_instances[i](sample[i])
                            first_3 = IMFs[:3,:]
                            if IMFs.shape[0] < 3:
                                padding = np.zeros((3 - IMFs.shape[0], IMFs.shape[1]))
                                first_3 = np.vstack((IMFs, padding))
                            sample_IMFs.append(first_3)
                        
                        batch_results.append(np.array(sample_IMFs))
                        processed_count += 1
                        
                    except Exception as e:
                        logging.error(f"Error processing sample {index}: {e}")
                        batch_results.append(np.zeros((3, 3, 6000), dtype=np.float32))
                        processed_count += 1
                
                # 写入批次
                for i, result in enumerate(batch_results):
                    dset[batch_start + i] = result
                
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
