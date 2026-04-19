# -*- coding: utf-8 -*-

import os
import copy
import sys

import numpy as np
import torch
import pickle
from torch.utils.data import Dataset


class Graph_bulider(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def get_Graph_data(data_path):
    with open(data_path,"rb") as f:
        data = pickle.load(f)

    dataset = Graph_bulider(data)
    return dataset

def get_Moe_data(data_path):

    with open(data_path, "rb") as f:
        data = pickle.load(f)

    filtered = []
    for sample in data:
        flag = sample["microb_flag"]

        if isinstance(flag, torch.Tensor):
            flag_value = flag.item()  # 转为 python bool
        else:
            flag_value = bool(flag)

        if flag_value:
            filtered.append(sample)
    dataset = Graph_bulider(filtered)
    return dataset

def infer_microbe_dim(dataset):
    for i in range(len(dataset)):
        if dataset[i]["microb_flag"] == 1:
            return int(dataset[i]["microb_feat"].shape[-1])

# HyMMSurv
def collate_multi_HyMMSurv(batch):
    assert len(batch) == 1, f"collate_multi_M2Surv only supports batch_size == 1, but got {len(batch)}"

    item = batch[0]

    return {
        "sample_name": [item["sample_name"]],

        # ---- Microbe ----
        "microb_feat": item["microb_feat"].unsqueeze(0),
        "microb_flag": torch.tensor(
            [item["microb_flag"]],
            dtype=torch.bool
        ),

        # ---- genomic ----  -> [B, 1, D]
        # "genomic_ts": item["genomic_ts"].unsqueeze(0),
        # "genomic_onco": item["genomic_onco"].unsqueeze(0),
        # "genomic_kinase": item["genomic_kinase"].unsqueeze(0),
        # "genomic_diff": item["genomic_diff"].unsqueeze(0),
        # "genomic_tf": item["genomic_tf"].unsqueeze(0),
        # "genomic_cytokine": item["genomic_cytokine"].unsqueeze(0),

        # ---- Survival ----
        "task_time": torch.tensor([item["task_time"]], dtype=torch.float32),
        "label": torch.tensor([item["label"]], dtype=torch.long),
        "risk_label": torch.tensor([item["risk_label"]], dtype=torch.long),

        # ---- muti_Graph 原样返回（图结构不做 batch 合并） ----
        "muti_Graph": item["muti_Graph"],
        "coords_20x": item["coords_20x"],
        "coords_10x": item["coords_10x"]

    }

def select_topN_feature_and_coord(feat, coord, topN):
    """
    截取前 N 个 patch 和坐标
    feat:  [P, D]
    coord: [P, C]
    topN:  int

    return:
        feat_top:  [topN, D]
        coord_top: [topN, C]
    """

    # 若 patch 数不足 N，直接截到实际长度
    n = min(topN, feat.shape[0])

    feat_top = feat[:n]
    coord_top = coord[:n]

    return feat_top, coord_top

def check_empty_genomic_fields(data):
    # 检查样本的基因特征缺失的情况
    genomic_keys = [
        'genomic_ts',
        'genomic_onco',
        'genomic_kinase',
        'genomic_diff',
        'genomic_tf',
        'genomic_cytokine'
    ]

    results = {k: [] for k in genomic_keys}

    for i, sample in enumerate(data):
        for k in genomic_keys:
            v = sample.get(k, None)

            is_empty = False

            # 1 None
            if v is None:
                is_empty = True

            # 2 numpy array
            elif isinstance(v, np.ndarray):
                if v.size == 0 or np.all(np.isnan(v)) or np.all(v == 0):
                    is_empty = True

            # 3 torch tensor
            elif torch.is_tensor(v):
                if v.numel() == 0 or torch.all(v == 0):
                    is_empty = True

            # 4 list / dict / str
            elif hasattr(v, "__len__"):
                if len(v) == 0:
                    is_empty = True

            if is_empty:
                results[k].append(i)

    for k in genomic_keys:
        print(f"{k}: {len(results[k])} empty samples")

    return results
