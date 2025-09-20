import json
import time
from pathlib import Path
import subprocess
import argparse
import signal
import sys
from datetime import datetime

class Monitor:
    def __init__(self, total_workers, cache_dir, output_dir, status_dir, log_dir):
        self.total_workers = total_workers
        self.cache_dir = Path(cache_dir)
        self.output_dir = Path(output_dir)
        self.status_dir = Path(status_dir)
        self.log_dir = Path(log_dir)
        self.processes = {}
        self.running = True
        
        # 创建目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.status_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        print("Received termination signal, stopping all workers...")
        self.running = False
        self.stop_all_workers()
    
    def start_worker(self, worker_id):
        """启动单个worker进程"""
        # Worker有自己的logging配置，不需要重定向stdout
        process = subprocess.Popen([
            sys.executable, 'worker_iquique.py',
            '--worker-id', str(worker_id),
            '--cache-dir', str(self.cache_dir),
            '--output-dir', str(self.output_dir),
            '--status-dir', str(self.status_dir)
        ])
        self.processes[worker_id] = process
        print(f"Started worker {worker_id} (PID: {process.pid})")
    
    def stop_worker(self, worker_id):
        """停止单个worker进程"""
        if worker_id in self.processes:
            self.processes[worker_id].terminate()
            self.processes[worker_id].wait()
            del self.processes[worker_id]
            print(f"Stopped worker {worker_id}")
    
    def stop_all_workers(self):
        """停止所有worker进程"""
        for worker_id in list(self.processes.keys()):
            self.stop_worker(worker_id)
    
    def check_worker_status(self, worker_id):
        """检查worker状态"""
        status_file = self.status_dir / f'worker_{worker_id}.json'
        
        if not status_file.exists():
            return 'not_started'
        
        try:
            with open(status_file, 'r') as f:
                status = json.load(f)
            return status.get('status', 'unknown')
        except:
            return 'corrupted'
    
    def monitor_workers(self):
        """监控所有worker"""
        print(f"Starting monitor for {self.total_workers} workers...")
        
        # 启动所有worker
        for worker_id in range(self.total_workers):
            self.start_worker(worker_id)
            time.sleep(1)  # 避免同时启动太多进程
        
        # 监控循环
        while self.running:
            try:
                completed = 0
                running = 0
                failed = 0
                
                for worker_id in range(self.total_workers):
                    status = self.check_worker_status(worker_id)
                    
                    if status == 'completed':
                        completed += 1
                    elif status in ['running', 'terminated']:
                        running += 1
                    elif status == 'failed':
                        failed += 1
                        # 自动重启失败的worker
                        if worker_id not in self.processes:
                            print(f"Restarting failed worker {worker_id}")
                            self.start_worker(worker_id)
                
                # 显示进度
                print(f"\rProgress: {completed}/{self.total_workers} completed, "
                      f"{running} running, {failed} failed", end='')
                
                # 检查进程状态
                for worker_id in list(self.processes.keys()):
                    process = self.processes[worker_id]
                    if process.poll() is not None:  # 进程已结束
                        status = self.check_worker_status(worker_id)
                        if status != 'completed':
                            print(f"\nWorker {worker_id} died unexpectedly, restarting...")
                            self.stop_worker(worker_id)
                            self.start_worker(worker_id)
                
                if completed == self.total_workers:
                    print("\nAll workers completed successfully!")
                    break
                
                time.sleep(10)  # 每10秒检查一次
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Monitor error: {e}")
                time.sleep(30)
        
        self.stop_all_workers()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optimized STEAD EMD Processing Monitor')
    parser.add_argument('--total-workers', type=int, default=32, help='Total number of workers')
    parser.add_argument('--cache-dir', type=str, required=True, help='Directory with worker cache files')
    parser.add_argument('--output-dir', type=str, default='./output', help='Output directory')
    parser.add_argument('--status-dir', type=str, default='./status', help='Status directory')
    parser.add_argument('--log-dir', type=str, default='./logs', help='Log directory')
    
    args = parser.parse_args()
    
    monitor = Monitor(args.total_workers, args.cache_dir, args.output_dir, args.status_dir, args.log_dir)
    monitor.monitor_workers()
