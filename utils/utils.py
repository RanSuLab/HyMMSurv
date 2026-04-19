# -*- coding: utf-8 -*-

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import pickle
import random
import os
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.nn import LayerNorm
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Subset
from figs.KM_curve import merge_five_fold_risk_files, plot_km_curve_by_logrank


def reparameterize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


### Sets Seed for reproducible experiments.
def seed_torch(device,seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def infer_microbe_dim(dataset,args):
    for i in range(len(dataset)):
        _, microbe_feat, microb_mask, _, _, _, _ = dataset[i]
        if microb_mask == 1:
            return len(microbe_feat)  # 返回维度

def split_dataset_by_microb_mask(dataset):
    train_indices = []
    test_indices = []

    for idx in range(len(dataset)):
        _, _, microb_mask, _, _, _, _ = dataset[idx]
        if microb_mask.item() == 0:
            test_indices.append(idx)
        else:
            train_indices.append(idx)

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, test_dataset

def pretty_print_args(args, title="Arguments"):
    print("=" * 60)
    print(f"{title}")
    print("=" * 60)

    args_dict = vars(args)
    max_key_len = max(len(k) for k in args_dict.keys())

    for k in sorted(args_dict.keys()):
        v = args_dict[k]
        print(f"{k:<{max_key_len}} : {v}")

    print("=" * 60)

def plot_merged_KM_cur(folder_path):
    # 绘制每一折的risk数据病保存
    merge_five_fold_risk_files(folder_path, output_name="merged_risk.csv")
    plot_km_curve_by_logrank(folder_path + "/merged_risk.csv",
                             folder_path + "/merged_Log_rank_best_thread_km_curve.png"
                             )

def load_and_concat_pkls(dir_path):
    all_items = []

    for fname in os.listdir(dir_path):
        if not fname.endswith(".pkl"):
            continue

        fpath = os.path.join(dir_path, fname)
        # print(f"loading: {fpath}")

        with open(fpath, "rb") as f:
            data = pickle.load(f)

        # 常见情况：每个文件是 list[…]，直接 extend
        if isinstance(data, list):
            all_items.extend(data)

        # 兼容：如果是 dict，放入列表
        else:
            all_items.append(data)

    # print(f"total items: {len(all_items)}")
    return all_items


    # 加载先验原型
    # folder = '/root/disk4/fengyun/TCGA/code_project/Micriobiome_WSI/code/test_experiments/WSI_Mirc_PaMoE_VAE/prototypes/'
    # STAD = torch.load(folder+'STAD.pt')
    # print(f'STAD:{STAD}')
    #
    # COAD = torch.load(folder + 'COAD.pt')
    # print(f'COAD:{COAD}')
    #
    # ESCA = torch.load(folder + 'ESCA.pt')
    # print(f'ESCA:{ESCA}')

    STAD_folder="/root/disk4/fengyun/TCGA/code_project/Micriobiome_WSI/code/test_experiments/WSI_Mirc_PAGHC_GNN_Sur/results/READ-OS-Sur_GNN_hypergraph/READ/WGS/"
    label_file = "/root/disk4/fengyun/TCGA/code_project/Micriobiome_WSI/dataset/TCMA/prognosis_labels/READ_prognosis_labels.csv"
    microb_file ="/root/disk4/fengyun/TCGA/code_project/Micriobiome_WSI/dataset/TCMA/abundance/WGS/bacteria.unambiguous.decontam.tissue.sample.rpm.relabund.txt"
    pathology_dir_20x = "/root/disk4/fengyun/TCGA/code_project/Micriobiome_WSI/dataset/TCGA-WSI/READ/trident_processed_10/10x_256px_0px_overlap/features_uni_v2"
    pathology_dir_10x = "/root/disk4/fengyun/TCGA/code_project/Micriobiome_WSI/dataset/TCGA-WSI/READ/trident_processed/20x_256px_0px_overlap/features_uni_v2"
    merge_representations_into_dataset(
        STAD_folder,
        label_file,
        microb_file,
        pathology_dir_20x,
        pathology_dir_10x
        )