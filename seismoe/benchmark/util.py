"""
This script offers general functionality required in multiple places.
"""

import numpy as np
import pandas as pd
import os
import logging
import packaging
import pytorch_lightning as pl
import seisbench.generate as sbg

def load_best_model(model_cls, weights, version):
    """
    Determines the model with lowest validation loss from the csv logs and loads it

    :param model_cls: Class of the lightning module to load
    :param weights: Path to weights as in cmd arguments
    :param version: String of version file
    :return: Instance of lightning module that was loaded from the best checkpoint
    """
    metrics = pd.read_csv(weights / version / "metrics.csv")

    idx = np.nanargmin(metrics["val_loss"])
    min_row = metrics.iloc[idx]

    #  For default checkpoint filename, see https://github.com/Lightning-AI/lightning/pull/11805
    #  and https://github.com/Lightning-AI/lightning/issues/16636.
    #  For example, 'epoch=0-step=1.ckpt' means the 1st step has finish, but the 1st epoch hasn't
    checkpoint = f"epoch={min_row['epoch']:.0f}-step={min_row['step']+1:.0f}.ckpt"

    # For default save path of checkpoints, see https://github.com/Lightning-AI/lightning/pull/12372
    checkpoint_path = weights / version / "checkpoints" / checkpoint
    print(f"Loading checkpoint {checkpoint_path} with val_loss {min_row['val_loss']:.4f}")
    return model_cls.load_from_checkpoint(checkpoint_path)


default_workers = os.getenv("BENCHMARK_DEFAULT_WORKERS", None)
if default_workers is None:
    logging.warning(
        "BENCHMARK_DEFAULT_WORKERS not set. "
        "Will use 16 workers if not specified otherwise in configuration."
    )
    default_workers = 8
else:
    default_workers = int(default_workers)


class CustomGenericGenerator(sbg.GenericGenerator):
    def _populate_state_dict(self, idx):
        # Call the parent method to get the base state_dict
        state_dict = super()._populate_state_dict(idx)
        
        # Extract emd_features from metadata and make it a top-level key
        x_data, x_meta = state_dict["X"]
        if "emd_stats" in x_meta:
            # emd_features is stored as (data, metadata) tuple, matching "X"
            state_dict["emd_stats"] = (x_meta["emd_stats"], x_meta)
        
        return state_dict
    
    def _clean_state_dict(self, state_dict):
        # Preserve both "X" and "emd_features" as data-only (strip metadata)
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            cleaned_state_dict[k] = v[0]  # v[0] is the data array
        return cleaned_state_dict

class CustomSteeredGenerator(sbg.SteeredGenerator):
    def _populate_state_dict(self, idx):
        # 调用父类方法获取基础 state_dict
        state_dict = super()._populate_state_dict(idx)
        
        # 提升 emd_features 为顶级键（如果在 metadata 中）
        x_data, x_meta = state_dict["X"]
        if "emd_stats" in x_meta:
            state_dict["emd_stats"] = (x_meta["emd_stats"], x_meta)
        
        return state_dict
    
    def _clean_state_dict(self, state_dict):
        # 保留 "X" 和 "emd_features" 作为数据
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            cleaned_state_dict[k] = v[0] if isinstance(v, tuple) else v
        return cleaned_state_dict