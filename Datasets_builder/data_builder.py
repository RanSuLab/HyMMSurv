# -*- coding: UTF-8 -*-
import os
import h5py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import Counter
from torch.utils.data import Dataset

import re
from collections import defaultdict
import pickle

def load_h5_cached(path):
    with h5py.File(path, 'r') as f:
        # 特征
        feat = torch.tensor(f['features'][:], dtype=torch.float32)
        # 坐标
        coords = torch.tensor(f['coords'][:], dtype=torch.float32)
    return feat, coords

def build_binary_risk_labels(survival_times):
   
    survival_times = np.array(survival_times, dtype=float)
    median_time = np.median(survival_times)
    return (survival_times <= median_time).astype(int)

def data_loder(label_file, microb_file, pathology_dir_20x, pathology_dir_10x, data_save,task_col='OS'):

    
    label_df = pd.read_csv(label_file)

    if task_col not in label_df.columns:
        raise ValueError(f"no task {task_col}")

    task_time = f"{task_col}.time"

    
    microb_feature = pd.read_csv(microb_file, sep='\t')
    numeric_cols = microb_feature.select_dtypes(exclude=['object']).columns
    microb_feature[numeric_cols] = microb_feature[numeric_cols].astype(np.float32)

    microb_dim = microb_feature.shape[0]

    pathology_dir_20x = pathology_dir_20x
    caseid_to_h5_20x = {}

    for fname in os.listdir(pathology_dir_20x):
        if fname.endswith('.h5'):
            case_id = fname[:12]
            full_path = os.path.join(pathology_dir_20x, fname)
            caseid_to_h5_20x[case_id] = full_path

    # 10x
    pathology_dir_10x = pathology_dir_10x
    caseid_to_h5_10x = {}

    for fname in os.listdir(pathology_dir_10x):
        if fname.endswith('.h5'):
            case_id = fname[:12]
            full_path = os.path.join(pathology_dir_10x, fname)
            caseid_to_h5_10x[case_id] = full_path

    samples_meta = []
    lock_num = 0

    for _, row in label_df.iterrows():
        sample_name = row.iloc[0]
        label_value = row[task_col]
        task_time_value = row[task_time]
        if pd.isna(label_value) or str(label_value).strip() == '#N/A':
            continue

        if (
            pd.isna(task_time_value)
            or str(task_time_value).strip() == '#N/A'
            or float(task_time_value) == 0
        ):
            continue


        has_20x = sample_name in caseid_to_h5_20x
        has_10x = sample_name in caseid_to_h5_10x

     
        if not has_20x:
            continue
        if not has_10x:
            continue

        print(f'正在处理样本：{sample_name}')

        path_20x_file = caseid_to_h5_20x.get(sample_name, None)
        path_20x, coords_20x = load_h5_cached(path_20x_file)
        path_10x_file = caseid_to_h5_10x.get(sample_name, None)
        path_10x, coords_10x = load_h5_cached(path_10x_file)

    
        kept_cols = []
        matching_cols = [
            col for col in microb_feature.columns[1:]
            if '-'.join(col.split('-')[:3]) == sample_name
        ]

        if matching_cols:
            for col in matching_cols:
                parts = col.split('-')
                sample_type = parts[3][:2]

                if sample_type in ['01', '02', '03', '06']:
                    kept_cols.append(col)

    
        if len(kept_cols) == 0:
            microb_feat = torch.zeros(microb_dim)
            microb_flag = 0
            lock_num += 1
        else:
            feats = microb_feature[matching_cols].values
            if len(kept_cols) == 1:
                microb_feat = torch.from_numpy(feats[:, 0]).float()
            else:
                microb_feat = torch.from_numpy(feats.mean(axis=1)).float()
            microb_flag = 1

    
        sample_one = {
            "sample_name": sample_name,
            "microb_feat": microb_feat,
            "microb_flag": torch.tensor(microb_flag, dtype=torch.bool),
            "path_20x": path_20x,
            "coords_20x": coords_20x,
            "path_20x_flag": torch.tensor(int(has_20x), dtype=torch.bool),
            "path_10x": path_10x,
            "coords_10x": coords_10x,
            "path_10x_flag": torch.tensor(int(has_10x), dtype=torch.bool),
            "genes_flag": torch.tensor(genes_flag, dtype=torch.bool),
            "label": torch.tensor(bool(label_value), dtype=torch.long),
            "task_time": torch.tensor(float(task_time_value), dtype=torch.float32),
        }
 
        samples_meta.append(sample_one)

    all_times = [sample["task_time"].item() for sample in samples_meta]
    risk_labels = build_binary_risk_labels(all_times)
    for sample, risk_label in zip(samples_meta, risk_labels):
        sample["risk_label"] = torch.tensor(risk_label, dtype=torch.long)

    with open(data_save, "wb") as f:
        pickle.dump(samples_meta, f)


    
if __name__ == "__main__":
    label_file = './dataset/prognosis_labels/STAD_prognosis_labels.csv'
    micr_file = './dataset/abundance/bacteria.unambiguous.decontam.tissue.sample.rpm.relabund.txt'
    path_dir20x = './dataset/TCGA-Features/STAD/trident_processed_20x/20x_256px_0px_overlap/'
    path_dir10x = './dataset/TCGA-Features/STAD/trident_processed_10x/10x_256px_0px_overlap/'
    data_save = './dataset/dataset_orignal/STAD_OS_WGS.pkl'
    task_col = 'OS'
    data_loder(label_file, micr_file, path_dir20x, path_dir10x, data_save, task_col)
    


 

