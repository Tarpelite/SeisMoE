"""
This file contains functionality related to data.
"""

from importlib.metadata import metadata
import seisbench.data as sbd
import seisbench
from pathlib import Path
import h5py
import numpy as np

# Merged EMD data configuration
MERGED_GEOFON_EMD_FILE_PATH = Path("/mnt/data/ICASSP2026/emd_feature/geofon/emd.hdf5")
MERGED_ETHZ_EMD_FILE_PATH = Path("/mnt/data/ICASSP2026/emd_feature/ethz/emd.hdf5")
MERGED_Iquique_EMD_FILE_PATH = Path("/mnt/data/ICASSP2026/emd_feature/iquique/emd.hdf5")



def get_emd_stats(emd_features):
    """
    Compute statistics from EMD features using NumPy.
    Args:
        emd_features: NumPy array of shape (batch, channels, imfs, time)
    Returns:
        stats: NumPy array of shape (batch, channels * imfs * 4)
    """
    # Reshape for easier computation: (batch, 9, time_length)
    b, c, i, t = emd_features.shape
    emd_flat = emd_features.reshape(b, c * i, t)

    mean = np.mean(emd_flat, axis=-1)
    std = np.std(emd_flat, axis=-1)
    
    # For skew and kurtosis, we need to be careful with dimensions
    mean_expanded = np.expand_dims(mean, axis=-1)
    std_expanded = np.expand_dims(std, axis=-1)
    std_expanded = np.clip(std_expanded, 1e-6, None)  # Clamp to avoid division by zero
    
    skew = np.mean(((emd_flat - mean_expanded) / std_expanded) ** 3, axis=-1)
    kurt = np.mean(((emd_flat - mean_expanded) / std_expanded) ** 4, axis=-1)

    # Concatenate features: (batch, 9, 4) -> (batch, 36)
    stats = np.stack([mean, std, skew, kurt], axis=-1)
    return stats.reshape(b, -1).squeeze()



class GEOFONWithEMD(sbd.GEOFON):
    """
    GEOFON dataset with EMD data.
    EMD特征的顺序与原始数据集完全一致，因此可以直接通过索引访问。
    """

    def __init__(self, **kwargs):
        print("Initializing GEOFONWithEMD dataset...")
        super().__init__(**kwargs)
        
        if not MERGED_GEOFON_EMD_FILE_PATH.exists():
            raise FileNotFoundError(f"EMD file not found at {MERGED_GEOFON_EMD_FILE_PATH}")
        
        # 打开EMD文件
        # self.emd_file_handle = h5py.File(MERGED_EMD_FILE_PATH, 'r')
        self.emd_file_path = MERGED_GEOFON_EMD_FILE_PATH
        # self.emd_file_handle = None
        self.emd_data = {}
        with h5py.File(MERGED_GEOFON_EMD_FILE_PATH, 'r') as f:
            for block_name in f['IMFs'].keys():
                self.emd_data[block_name] = f['IMFs'][block_name][:]  # 加载为 NumPy array

        self.imf_nums = 3
        self.stat_dim = 4
        
        # 检查是否有IMFs组
        # if 'IMFs' not in self.emd_file_handle:
        #     raise KeyError("Cannot find the 'IMFs' group in EMD file.")
        
    
        # self._check_completeness()
        print("GEOFONWithEMD initialized successfully.")

    def get_waveforms(self, idx, **kwargs):
        # 获取原始波形和 metadata（批量）
        original_waveforms, metadata = super().get_waveforms(idx, **kwargs)
        
        # 为每个样本添加 EMD 特征
        for i, sample_idx in enumerate(idx):
            # 获取当前样本的 metadata 行
            metadata_row = self.metadata.iloc[sample_idx]
            trace_name = metadata_row["trace_name"]
            
            # 解析 trace_name（与 get_sample 相同）
            if "$" in trace_name:
                block_name, location_str = trace_name.split("$")
            else:
                block_name, location_str = trace_name, ":"
            locations = self._parse_location(location_str)
            
            # 加载 EMD 特征
            try:
                emd_feature = self._get_emd_feature(block_name, locations)
                emd_feature = np.expand_dims(emd_feature, 0).astype(np.float32)  # add batch dimension
                emd_stats = get_emd_stats(emd_feature)  # 计算统计特征
                # print(type(emd_feature), emd_feature.shape)
                metadata['emd_stats'][i] = emd_stats
            except Exception as e:
                print(f"Warning: Could not load EMD feature for idx {sample_idx}: {e}")
                # 创建 dummy 数据，形状匹配波形时间维度
                time_length = original_waveforms.shape[-1]
                metadata['emd_stats'][i] = self._create_dummy_emd(self.stat_dim * self.imf_nums * 3)  # 3 channels, 3 IMFs, 4 stats each
        
        return original_waveforms, metadata


    def _check_completeness(self):
        """构建EMD数据的累积索引，用于快速定位"""
        emd_cumulative_indices = []
        # emd_dataset_names = []
        cumulative_count = 0
        
        # 按顺序读取所有EMD数据集
        all_datasets = list(self.emd_data['IMFs'].keys())
        # print(all_datasets)
        # # 确保按正确顺序排列（如果数据集名称有特定顺序要求）
        # all_datasets.sort()
        
        for dataset_name in all_datasets:
            dataset = self.emd_data['IMFs'][dataset_name]
            num_samples = dataset.shape[0]
            
            # self.emd_dataset_names.append(dataset_name)
            emd_cumulative_indices.append(cumulative_count + num_samples)
            cumulative_count += num_samples
        
        self.total_emd_samples = cumulative_count
        
        # 验证EMD样本数量与元数据数量是否一致
        if self.total_emd_samples != len(self.metadata):
            print(f"Warning: EMD samples ({self.total_emd_samples}) != metadata samples ({len(self.metadata)})")
        else:
            print(f"EMD data verification passed: {self.total_emd_samples} samples")

    def get_sample(self, idx, sampling_rate=None):
        """重写get_sample方法，添加EMD特征"""
        waveform, metadata = super().get_sample(idx, sampling_rate)
        # print(metadata['trace_name'])
        trace_name = metadata['trace_name']
        if trace_name.find("$") != -1:
            # 例如: "bucket0$0,0:3,0:8000"
            block_name, location = trace_name.split("$")  # block_name="bucket0", location="0,0:3,0:8000"
        else:
            # 例如: "gfz2011yrrp_IU.OTAV.00.BH"
            block_name, location = trace_name, ":"
        # print(f"Block: {block_name}, Location: {location}")
        locations = super()._parse_location(location)
        # print(f"Parsed locations: {locations}")
        # print(f"length of locations: {len(locations)}")
        
        # 获取对应的EMD特征
        try:
            emd_feature = self._get_emd_feature(block_name, locations)
            emd_feature = np.expand_dims(emd_feature, 0).astype(np.float32)  # add batch dimension
            emd_stats = get_emd_stats(emd_feature)  # 计算统计特征
                # print(type(emd_feature), emd_feature.shape)
            metadata['emd_stats'] = emd_stats
        except Exception as e:
            print(f"Warning: Could not load EMD feature for idx {idx}, bock {block_name} at {locations}: {e}")
            # 如果加载失败，创建零填充的EMD特征
            metadata['emd_stats'] = self._create_dummy_emd(self.stat_dim * self.imf_nums * 3)  # 3 channels, 3 IMFs, 4 stats each
        
        return waveform, metadata

    def _get_emd_feature(self, block_name, locations):
        """获取指定索引的EMD特征"""

        # handle = self._get_file_handle()
        block = self.emd_data[block_name]

        if len(locations) == 3:
            emd_data = block[locations[0]]
        else:
            emd_data = block[0]
        # 形状应为 (3, 3, time_length)
        # print(f"Loaded EMD data shape: {emd_data.shape}")
        return emd_data.astype(np.float32)
    
    def _create_dummy_emd(self, all_dim):
        """创建零填充的EMD特征"""
        return np.zeros(all_dim, dtype=np.float32)
    
    def __getitem__(self, idx):
        """
        Overrides the parent __getitem__. It fetches the original sample
        and then adds the corresponding EMD feature as a top-level key.
        """
        # 1. Get the original sample dictionary from the parent class.
        # This dictionary contains {'X': (waveform, metadata), **metadata}.
        sample_dict = super().__getitem__(idx)

        # 2. The information needed to slice the EMD data is in the original metadata row.
        metadata_row = self.metadata.iloc[idx]
        trace_name = metadata_row["trace_name"]

        # 3. Parse the trace_name to get block and location info
        if "$" in trace_name:
            block_name, location_str = trace_name.split("$")
        else:
            block_name, location_str = trace_name, ":"
        
        locations = self._parse_location(location_str)
        
        # 4. Load the corresponding EMD feature
        try:
            # handle = self._get_file_handle()
            block = self.emd_data['IMFs'][block_name]
            
            # EMD data shape: (samples, channels, imfs, time_length)
            # locations[0]: sample index in block
            # locations[2]: time slice
            if len(locations) != 3:
                emd_feature = block[locations[0]]
            else:
                emd_feature = block[0]
            emd_feature = np.expand_dims(emd_feature, 0).astype(np.float32)  # add batch dimension
            emd_stats = get_emd_stats(emd_feature)  # 计算统计特征
                # print(type(emd_feature), emd_feature.shape)
            # 5. Add 'emd_features' as a new top-level key to the dictionary
            sample_dict['emd_stats'] = (emd_stats.astype(np.float32), sample_dict['X'][1])

        except Exception as e:
            print(f"Warning: Could not load EMD feature for idx {idx} (trace: {trace_name}). Error: {e}")
            # If loading fails, create a dummy feature.
            # The shape must match the waveform's time dimension.
            # time_length = sample_dict['X'][0].shape[-1]
            dummy_emd = np.zeros((1, self.stat_dim * self.imf_nums * 3), dtype=np.float32)
            sample_dict['emd_stats'] = (dummy_emd, sample_dict['X'][1])
        
        return sample_dict

    @property
    def _cache_name(self):
        """使用父类的缓存名避免重复下载"""
        return "GEOFON"
    
    @classmethod 
    def _name_internal(cls):
        """返回父类的数据集名"""
        return "GEOFON"

class ETHZWithEMD(sbd.ETHZ):
    """
    ETHZ dataset with EMD data.
    EMD特征的顺序与原始数据集完全一致，因此可以直接通过索引访问。
    """

    def __init__(self, **kwargs):
        print("Initializing ETHZWithEMD dataset...")
        super().__init__(**kwargs)
        
        if not MERGED_ETHZ_EMD_FILE_PATH.exists():
            raise FileNotFoundError(f"EMD file not found at {MERGED_ETHZ_EMD_FILE_PATH}")
        
        # 打开EMD文件
        # self.emd_file_handle = h5py.File(MERGED_EMD_FILE_PATH, 'r')
        self.emd_file_path = MERGED_ETHZ_EMD_FILE_PATH
        # self.emd_file_handle = None
        self.emd_data = {}
        with h5py.File(MERGED_ETHZ_EMD_FILE_PATH, 'r') as f:
            for block_name in f['IMFs'].keys():
                self.emd_data[block_name] = f['IMFs'][block_name][:]  # 加载为 NumPy array

        self.imf_nums = 3
        self.stat_dim = 4
        
        # 检查是否有IMFs组
        # if 'IMFs' not in self.emd_file_handle:
        #     raise KeyError("Cannot find the 'IMFs' group in EMD file.")
        
    
        # self._check_completeness()
        print("ETHZWithEMD initialized successfully.")

    def get_waveforms(self, idx, **kwargs):
        # 获取原始波形和 metadata（批量）
        original_waveforms, metadata = super().get_waveforms(idx, **kwargs)
        
        # 为每个样本添加 EMD 特征
        for i, sample_idx in enumerate(idx):
            # 获取当前样本的 metadata 行
            metadata_row = self.metadata.iloc[sample_idx]
            trace_name = metadata_row["trace_name"]
            
            # 解析 trace_name（与 get_sample 相同）
            if "$" in trace_name:
                block_name, location_str = trace_name.split("$")
            else:
                block_name, location_str = trace_name, ":"
            locations = self._parse_location(location_str)
            
            # 加载 EMD 特征
            try:
                emd_feature = self._get_emd_feature(block_name, locations)
                emd_feature = np.expand_dims(emd_feature, 0).astype(np.float32)  # add batch dimension
                emd_stats = get_emd_stats(emd_feature)  # 计算统计特征
                # print(type(emd_feature), emd_feature.shape)
                metadata['emd_stats'][i] = emd_stats
            except Exception as e:
                print(f"Warning: Could not load EMD feature for idx {sample_idx}: {e}")
                # 创建 dummy 数据，形状匹配波形时间维度
                time_length = original_waveforms.shape[-1]
                metadata['emd_stats'][i] = self._create_dummy_emd(self.stat_dim * self.imf_nums * 3)  # 3 channels, 3 IMFs, 4 stats each
        
        return original_waveforms, metadata


    def _check_completeness(self):
        """构建EMD数据的累积索引，用于快速定位"""
        emd_cumulative_indices = []
        # emd_dataset_names = []
        cumulative_count = 0
        
        # 按顺序读取所有EMD数据集
        all_datasets = list(self.emd_data['IMFs'].keys())
        # print(all_datasets)
        # # 确保按正确顺序排列（如果数据集名称有特定顺序要求）
        # all_datasets.sort()
        
        for dataset_name in all_datasets:
            dataset = self.emd_data['IMFs'][dataset_name]
            num_samples = dataset.shape[0]
            
            # self.emd_dataset_names.append(dataset_name)
            emd_cumulative_indices.append(cumulative_count + num_samples)
            cumulative_count += num_samples
        
        self.total_emd_samples = cumulative_count
        
        # 验证EMD样本数量与元数据数量是否一致
        if self.total_emd_samples != len(self.metadata):
            print(f"Warning: EMD samples ({self.total_emd_samples}) != metadata samples ({len(self.metadata)})")
        else:
            print(f"EMD data verification passed: {self.total_emd_samples} samples")

    def get_sample(self, idx, sampling_rate=None):
        """重写get_sample方法，添加EMD特征"""
        waveform, metadata = super().get_sample(idx, sampling_rate)
        # print(metadata['trace_name'])
        trace_name = metadata['trace_name']
        if trace_name.find("$") != -1:
            # 例如: "bucket0$0,0:3,0:8000"
            block_name, location = trace_name.split("$")  # block_name="bucket0", location="0,0:3,0:8000"
        else:
            # 例如: "gfz2011yrrp_IU.OTAV.00.BH"
            block_name, location = trace_name, ":"
        # print(f"Block: {block_name}, Location: {location}")
        locations = super()._parse_location(location)
        # print(f"Parsed locations: {locations}")
        # print(f"length of locations: {len(locations)}")
        
        # 获取对应的EMD特征
        try:
            emd_feature = self._get_emd_feature(block_name, locations)
            emd_feature = np.expand_dims(emd_feature, 0).astype(np.float32)  # add batch dimension
            emd_stats = get_emd_stats(emd_feature)  # 计算统计特征
                # print(type(emd_feature), emd_feature.shape)
            metadata['emd_stats'] = emd_stats
        except Exception as e:
            print(f"Warning: Could not load EMD feature for idx {idx}, bock {block_name} at {locations}: {e}")
            # 如果加载失败，创建零填充的EMD特征
            metadata['emd_stats'] = self._create_dummy_emd(self.stat_dim * self.imf_nums * 3)  # 3 channels, 3 IMFs, 4 stats each
        
        return waveform, metadata

    def _get_emd_feature(self, block_name, locations):
        """获取指定索引的EMD特征"""

        # handle = self._get_file_handle()
        block = self.emd_data[block_name]

        if len(locations) == 3:
            emd_data = block[locations[0]]
        else:
            emd_data = block[0]
        # 形状应为 (3, 3, time_length)
        # print(f"Loaded EMD data shape: {emd_data.shape}")
        return emd_data.astype(np.float32)
    
    def _create_dummy_emd(self, all_dim):
        """创建零填充的EMD特征"""
        return np.zeros(all_dim, dtype=np.float32)
    
    def __getitem__(self, idx):
        """
        Overrides the parent __getitem__. It fetches the original sample
        and then adds the corresponding EMD feature as a top-level key.
        """
        # 1. Get the original sample dictionary from the parent class.
        # This dictionary contains {'X': (waveform, metadata), **metadata}.
        sample_dict = super().__getitem__(idx)

        # 2. The information needed to slice the EMD data is in the original metadata row.
        metadata_row = self.metadata.iloc[idx]
        trace_name = metadata_row["trace_name"]

        # 3. Parse the trace_name to get block and location info
        if "$" in trace_name:
            block_name, location_str = trace_name.split("$")
        else:
            block_name, location_str = trace_name, ":"
        
        locations = self._parse_location(location_str)
        
        # 4. Load the corresponding EMD feature
        try:
            # handle = self._get_file_handle()
            block = self.emd_data['IMFs'][block_name]
            
            # EMD data shape: (samples, channels, imfs, time_length)
            # locations[0]: sample index in block
            # locations[2]: time slice
            if len(locations) != 3:
                emd_feature = block[locations[0]]
            else:
                emd_feature = block[0]
            emd_feature = np.expand_dims(emd_feature, 0).astype(np.float32)  # add batch dimension
            emd_stats = get_emd_stats(emd_feature)  # 计算统计特征
                # print(type(emd_feature), emd_feature.shape)
            # 5. Add 'emd_features' as a new top-level key to the dictionary
            sample_dict['emd_stats'] = (emd_stats.astype(np.float32), sample_dict['X'][1])

        except Exception as e:
            print(f"Warning: Could not load EMD feature for idx {idx} (trace: {trace_name}). Error: {e}")
            # If loading fails, create a dummy feature.
            # The shape must match the waveform's time dimension.
            # time_length = sample_dict['X'][0].shape[-1]
            dummy_emd = np.zeros(self.stat_dim * self.imf_nums * 3, dtype=np.float32)
            sample_dict['emd_stats'] = (dummy_emd, sample_dict['X'][1])
        
        return sample_dict

    @property
    def _cache_name(self):
        """使用父类的缓存名避免重复下载"""
        return "ETHZ"
    
    @classmethod 
    def _name_internal(cls):
        """返回父类的数据集名"""
        return "ETHZ"


class IquiqueWithEMD(sbd.Iquique):
    """
    Iquique dataset with EMD data.
    EMD特征的顺序与原始数据集完全一致，因此可以直接通过索引访问。
    """

    def __init__(self, **kwargs):
        print("Initializing IquiqueWithEMD dataset...")
        super().__init__(**kwargs)
        
        if not MERGED_Iquique_EMD_FILE_PATH.exists():
            raise FileNotFoundError(f"EMD file not found at {MERGED_Iquique_EMD_FILE_PATH}")
        
        # 打开EMD文件
        # self.emd_file_handle = h5py.File(MERGED_EMD_FILE_PATH, 'r')
        self.emd_file_path = MERGED_Iquique_EMD_FILE_PATH
        # self.emd_file_handle = None
        self.emd_data = {}
        with h5py.File(MERGED_Iquique_EMD_FILE_PATH, 'r') as f:
            for block_name in f['IMFs'].keys():
                self.emd_data[block_name] = f['IMFs'][block_name][:]  # 加载为 NumPy array

        self.imf_nums = 3
        self.stat_dim = 4
        
        # 检查是否有IMFs组
        # if 'IMFs' not in self.emd_file_handle:
        #     raise KeyError("Cannot find the 'IMFs' group in EMD file.")
        
    
        # self._check_completeness()
        print("IquiqueWithEMD initialized successfully.")

    def get_waveforms(self, idx, **kwargs):
        # 获取原始波形和 metadata（批量）
        original_waveforms, metadata = super().get_waveforms(idx, **kwargs)
        
        # 为每个样本添加 EMD 特征
        for i, sample_idx in enumerate(idx):
            # 获取当前样本的 metadata 行
            metadata_row = self.metadata.iloc[sample_idx]
            trace_name = metadata_row["trace_name"]
            
            # 解析 trace_name（与 get_sample 相同）
            if "$" in trace_name:
                block_name, location_str = trace_name.split("$")
            else:
                block_name, location_str = trace_name, ":"
            locations = self._parse_location(location_str)
            
            # 加载 EMD 特征
            try:
                emd_feature = self._get_emd_feature(block_name, locations)
                emd_feature = np.expand_dims(emd_feature, 0).astype(np.float32)  # add batch dimension
                emd_stats = get_emd_stats(emd_feature)  # 计算统计特征
                # print(type(emd_feature), emd_feature.shape)
                metadata['emd_stats'][i] = emd_stats
            except Exception as e:
                print(f"Warning: Could not load EMD feature for idx {sample_idx}: {e}")
                # 创建 dummy 数据，形状匹配波形时间维度
                time_length = original_waveforms.shape[-1]
                metadata['emd_stats'][i] = self._create_dummy_emd(self.stat_dim * self.imf_nums * 3)  # 3 channels, 3 IMFs, 4 stats each
        
        return original_waveforms, metadata


    def _check_completeness(self):
        """构建EMD数据的累积索引，用于快速定位"""
        emd_cumulative_indices = []
        # emd_dataset_names = []
        cumulative_count = 0
        
        # 按顺序读取所有EMD数据集
        all_datasets = list(self.emd_data['IMFs'].keys())
        # print(all_datasets)
        # # 确保按正确顺序排列（如果数据集名称有特定顺序要求）
        # all_datasets.sort()
        
        for dataset_name in all_datasets:
            dataset = self.emd_data['IMFs'][dataset_name]
            num_samples = dataset.shape[0]
            
            # self.emd_dataset_names.append(dataset_name)
            emd_cumulative_indices.append(cumulative_count + num_samples)
            cumulative_count += num_samples
        
        self.total_emd_samples = cumulative_count
        
        # 验证EMD样本数量与元数据数量是否一致
        if self.total_emd_samples != len(self.metadata):
            print(f"Warning: EMD samples ({self.total_emd_samples}) != metadata samples ({len(self.metadata)})")
        else:
            print(f"EMD data verification passed: {self.total_emd_samples} samples")

    def get_sample(self, idx, sampling_rate=None):
        """重写get_sample方法，添加EMD特征"""
        waveform, metadata = super().get_sample(idx, sampling_rate)
        # print(metadata['trace_name'])
        trace_name = metadata['trace_name']
        if trace_name.find("$") != -1:
            # 例如: "bucket0$0,0:3,0:8000"
            block_name, location = trace_name.split("$")  # block_name="bucket0", location="0,0:3,0:8000"
        else:
            # 例如: "gfz2011yrrp_IU.OTAV.00.BH"
            block_name, location = trace_name, ":"
        # print(f"Block: {block_name}, Location: {location}")
        locations = super()._parse_location(location)
        # print(f"Parsed locations: {locations}")
        # print(f"length of locations: {len(locations)}")
        
        # 获取对应的EMD特征
        try:
            emd_feature = self._get_emd_feature(block_name, locations)
            emd_feature = np.expand_dims(emd_feature, 0).astype(np.float32)  # add batch dimension
            emd_stats = get_emd_stats(emd_feature)  # 计算统计特征
                # print(type(emd_feature), emd_feature.shape)
            metadata['emd_stats'] = emd_stats
        except Exception as e:
            print(f"Warning: Could not load EMD feature for idx {idx}, bock {block_name} at {locations}: {e}")
            # 如果加载失败，创建零填充的EMD特征
            metadata['emd_stats'] = self._create_dummy_emd(self.stat_dim * self.imf_nums * 3)  # 3 channels, 3 IMFs, 4 stats each
        
        return waveform, metadata

    def _get_emd_feature(self, block_name, locations):
        """获取指定索引的EMD特征"""

        # handle = self._get_file_handle()
        block = self.emd_data[block_name]

        if len(locations) == 3:
            emd_data = block[locations[0]]
        else:
            emd_data = block[0]
        # 形状应为 (3, 3, time_length)
        # print(f"Loaded EMD data shape: {emd_data.shape}")
        return emd_data.astype(np.float32)
    
    def _create_dummy_emd(self, all_dim):
        """创建零填充的EMD特征"""
        return np.zeros(all_dim, dtype=np.float32)
    
    def __getitem__(self, idx):
        """
        Overrides the parent __getitem__. It fetches the original sample
        and then adds the corresponding EMD feature as a top-level key.
        """
        # 1. Get the original sample dictionary from the parent class.
        # This dictionary contains {'X': (waveform, metadata), **metadata}.
        sample_dict = super().__getitem__(idx)

        # 2. The information needed to slice the EMD data is in the original metadata row.
        metadata_row = self.metadata.iloc[idx]
        trace_name = metadata_row["trace_name"]

        # 3. Parse the trace_name to get block and location info
        if "$" in trace_name:
            block_name, location_str = trace_name.split("$")
        else:
            block_name, location_str = trace_name, ":"
        
        locations = self._parse_location(location_str)
        
        # 4. Load the corresponding EMD feature
        try:
            # handle = self._get_file_handle()
            block = self.emd_data['IMFs'][block_name]
            
            # EMD data shape: (samples, channels, imfs, time_length)
            # locations[0]: sample index in block
            # locations[2]: time slice
            if len(locations) != 3:
                emd_feature = block[locations[0]]
            else:
                emd_feature = block[0]
            emd_feature = np.expand_dims(emd_feature, 0).astype(np.float32)  # add batch dimension
            emd_stats = get_emd_stats(emd_feature)  # 计算统计特征
                # print(type(emd_feature), emd_feature.shape)
            # 5. Add 'emd_features' as a new top-level key to the dictionary
            sample_dict['emd_stats'] = (emd_stats.astype(np.float32), sample_dict['X'][1])

        except Exception as e:
            print(f"Warning: Could not load EMD feature for idx {idx} (trace: {trace_name}). Error: {e}")
            # If loading fails, create a dummy feature.
            # The shape must match the waveform's time dimension.
            # time_length = sample_dict['X'][0].shape[-1]
            dummy_emd = np.zeros(self.stat_dim * self.imf_nums * 3, dtype=np.float32)
            sample_dict['emd_stats'] = (dummy_emd, sample_dict['X'][1])
        
        return sample_dict

    @property
    def _cache_name(self):
        """使用父类的缓存名避免重复下载"""
        return "Iquique"
    
    @classmethod 
    def _name_internal(cls):
        """返回父类的数据集名"""
        return "Iquique"

def get_dataset_by_name(name):
    """
    Resolve dataset name to class from seisbench.data.

    :param name: Name of dataset as defined in seisbench.data.
    :return: Dataset class from seisbench.data
    """
    print(f"Seisbench cache location: {seisbench.cache_root}")
    
    # if name == "GEOFONWithEMD":
    #     return GEOFONWithEMD
    # try:
    #     return sbd.__getattribute__(name)
    # except AttributeError:
    #     raise ValueError(f"Unknown dataset '{name}'.")
    try:
        match name:
            case "GEOFONWithEMD":
                return GEOFONWithEMD
            case "ETHZWithEMD":
                return ETHZWithEMD
            case "IquiqueWithEMD":
                return IquiqueWithEMD
            case _:
                return sbd.__getattribute__(name)
    except AttributeError:
        raise ValueError(f"Unknown dataset '{name}'.")