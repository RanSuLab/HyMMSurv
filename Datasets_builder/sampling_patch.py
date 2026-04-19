# -*- coding: utf-8 -*-
import torch
import os
from pathlib import Path
import random
import numpy as np
import pickle
import argparse
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 如果用到 GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 使 CuDNN 可复现（牺牲一点性能换确定性）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 没对patch坐标进行更新
    set_seed(42)
    # 采样指定数量的patch，并且同步修改超边索引
    path_20x = graph.path_20x
    path_10x = graph.path_10x
    path20_edge_index = graph.path20_edge_index
    path10_edge_index = graph.path10_edge_index
    share_edge = graph.share_edge

    # ========= 采样 20x patch =========
    N20 = path_20x.size(0)
    if N20 > num_patches:
        idx20 = np.random.choice(N20, num_patches, replace=False)
        idx20_set = set(idx20.tolist())
        path_20x = path_20x[idx20]
        # 过滤 20x 边
        e20 = path20_edge_index.cpu().numpy()
        mask20 = np.isin(e20[0], idx20) & np.isin(e20[1], idx20)
        e20 = e20[:, mask20]

        # 重映射 20x 节点 id
        map20 = {old: new for new, old in enumerate(idx20)}
        if e20.shape[1] > 0:
            e20[0] = np.vectorize(map20.get)(e20[0])
            e20[1] = np.vectorize(map20.get)(e20[1])

        edge_20x = torch.tensor(e20, dtype=torch.long)
    else:
        map20 = {i: i for i in range(N20)}
        idx20_set = set(range(N20))
        edge_20x = path20_edge_index

    # ========= 采样 10x patch =========
    N10 = path_10x.size(0)
    if N10 > num_patches:
        idx10 = np.random.choice(N10, num_patches, replace=False)
        idx10_set = set(idx10.tolist())

        path_10x = path_10x[idx10]

        # 过滤 10x 边
        e10 = path10_edge_index.cpu().numpy()
        mask10 = np.isin(e10[0], idx10) & np.isin(e10[1], idx10)
        e10 = e10[:, mask10]

        # 重映射 10x 节点 id
        map10 = {old: new for new, old in enumerate(idx10)}
        if e10.shape[1] > 0:
            e10[0] = np.vectorize(map10.get)(e10[0])
            e10[1] = np.vectorize(map10.get)(e10[1])

        edge_10x = torch.tensor(e10, dtype=torch.long)
    else:
        map10 = {i: i for i in range(N10)}
        idx10_set = set(range(N10))
        edge_10x = path10_edge_index

    # ========= 处理 share_edge =========
    if share_edge is not None:
        share = share_edge.cpu().numpy()

        # 只保留 同时在两个采样集合中的共享边
        mask_share = (
                np.isin(share[0], list(idx20_set)) &
                np.isin(share[1], list(idx10_set))
        )

        share = share[:, mask_share]

        # 映射两个倍率
        if share.shape[1] > 0:
            share[0] = np.vectorize(map20.get)(share[0])  # 20x
            share[1] = np.vectorize(map10.get)(share[1])  # 10x

        share_edge = torch.tensor(share, dtype=torch.long)

    # ========= 5. 回写 graph =========
    graph.path_20x = path_20x
    graph.path_10x = path_10x

    graph.path20_edge_index = edge_20x
    graph.path10_edge_index = edge_10x

    graph.share_edge = share_edge

    return graph

def sampling_patch_and_coords(graph,coords_20x, coords_10x,num_patches):
    set_seed(42)
    path_20x = graph.path_20x
    path_10x = graph.path_10x
    path20_edge_index = graph.path20_edge_index
    path10_edge_index = graph.path10_edge_index
    share_edge = graph.share_edge

    N20 = path_20x.size(0)
    if N20 > num_patches:
        idx20 = np.random.choice(N20, num_patches, replace=False)
        idx20_set = set(idx20.tolist())
        path_20x = path_20x[idx20]
        coords_20x_sampled = coords_20x[idx20]
        # 过滤 20x 边
        e20 = path20_edge_index.cpu().numpy()
        mask20 = np.isin(e20[0], idx20) & np.isin(e20[1], idx20)
        e20 = e20[:, mask20]

        # 重映射 20x 节点 id
        map20 = {old: new for new, old in enumerate(idx20)}
        if e20.shape[1] > 0:
            e20[0] = np.vectorize(map20.get)(e20[0])
            e20[1] = np.vectorize(map20.get)(e20[1])

        edge_20x = torch.tensor(e20, dtype=torch.long)
    else:
        map20 = {i: i for i in range(N20)}
        idx20_set = set(range(N20))
        edge_20x = path20_edge_index
        coords_20x_sampled = coords_20x

    N10 = path_10x.size(0)
    if N10 > num_patches:
        idx10 = np.random.choice(N10, num_patches, replace=False)
        idx10_set = set(idx10.tolist())

        path_10x = path_10x[idx10]
        coords_10x_sampled = coords_10x[idx10]

        e10 = path10_edge_index.cpu().numpy()
        mask10 = np.isin(e10[0], idx10) & np.isin(e10[1], idx10)
        e10 = e10[:, mask10]

        map10 = {old: new for new, old in enumerate(idx10)}
        if e10.shape[1] > 0:
            e10[0] = np.vectorize(map10.get)(e10[0])
            e10[1] = np.vectorize(map10.get)(e10[1])

        edge_10x = torch.tensor(e10, dtype=torch.long)
    else:
        map10 = {i: i for i in range(N10)}
        idx10_set = set(range(N10))
        edge_10x = path10_edge_index
        coords_10x_sampled = coords_10x

    # ========= share_edge =========
    if share_edge is not None:
        share = share_edge.cpu().numpy()

        mask_share = (
                np.isin(share[0], list(idx20_set)) &
                np.isin(share[1], list(idx10_set))
        )

        share = share[:, mask_share]

        if share.shape[1] > 0:
            share[0] = np.vectorize(map20.get)(share[0])  # 20x
            share[1] = np.vectorize(map10.get)(share[1])  # 10x

        share_edge = torch.tensor(share, dtype=torch.long)

    graph.path_20x = path_20x
    graph.path_10x = path_10x

    graph.path20_edge_index = edge_20x
    graph.path10_edge_index = edge_10x

    graph.share_edge = share_edge

    return graph, coords_20x_sampled, coords_10x_sampled

def main(args):
    data_path = args.data_path
    num_patches = args.num_patches
    file_name = Path(data_path).stem
    dir_path = os.path.dirname(data_path)
    save_path = os.path.join(dir_path, f'graph_sampling_{str(num_patches)}')
    os.makedirs(save_path, exist_ok=True)
    print(f'save_path:',save_path)

    with open(data_path, "rb") as f:
        data = pickle.load(f)
    for item in data:
        print('sample_name:', item['sample_name'])
        # G = wholegraph(item)
        graph = item["muti_Graph"]
        coords_20x = item["coords_20x"]
        coords_10x = item["coords_10x"]

        new_graph,coords_20x_sampled,coords_10x_sampled = sampling_patch_and_coords(graph,coords_20x, coords_10x,num_patches)
        item["muti_Graph"] = new_graph
        item["coords_20x"] = coords_20x_sampled
        item["coords_10x"] = coords_10x_sampled

    name = str(file_name)+'_'+str(num_patches)+".pkl"
    file_path = os.path.join(save_path, name)
    with open(file_path, "wb") as f:
        pickle.dump(data, f)
    print('Sampling ',num_patches,' patch saved：', file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type = str,required=True,
                        help='data_path')
    parser.add_argument('--num_patches', type=int, default=2000,
                        help='num_patches')
    args = parser.parse_args()
    main(args)
