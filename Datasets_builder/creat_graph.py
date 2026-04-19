import os
import argparse
import numpy as np
import nmslib
import torch
from torch_geometric.data import Data as geomData
from itertools import chain
from pathlib import Path

class Hnsw:
    def __init__(self, space='cosinesimil', index_params=None,
                 query_params=None):
        self.space = space 
        self.index_params = index_params
        self.query_params = query_params

    def fit(self, X):
        index_params = self.index_params
        if index_params is None:
            index_params = {'M': 12, 'post': 0, 'efConstruction': 400}

        query_params = self.query_params
        if query_params is None:
            query_params = {'ef': 90}

        index = nmslib.init(space=self.space, method='hnsw')
        index.addDataPointBatch(X)
        index.createIndex(index_params, False)
        index.setQueryTimeParams(query_params)

        self.index_ = index
        self.index_params_ = index_params
        self.query_params_ = query_params
        return self

    def query(self, vector, topn):
        indices, dist = self.index_.knnQuery(vector, k=topn)
        return indices

def wsi_hypergraph(sample_dict, radius):
    def process_slide(coords, current_node_num, radius):
        num_patches = coords.shape[0]
        model = Hnsw(space='l2')
        model.fit(coords)
        a = np.repeat(range(current_node_num, current_node_num + num_patches), radius - 1)
        b = np.fromiter(chain(*[model.query(coords[v_idx], topn=radius)[1:] for v_idx in range(num_patches)]), dtype=int) + current_node_num
        edge_spatial = torch.Tensor(np.stack([a, b])).type(torch.LongTensor)

        return edge_spatial

    # Single-magnification hyperedges.
    features_20x = sample_dict['path_20x']
    edge_spatial_20x = process_slide(sample_dict['coords_20x'], 0, radius)

    features_10x = sample_dict['path_10x']
    edge_spatial_10x = process_slide(sample_dict['coords_10x'], 0, radius)

    # Cross-magnification hyperedge
  
    whole_features_combined = np.concatenate([features_20x, features_10x], axis=0)
    model_shared_edges = Hnsw(space='l2')
    model_shared_edges.fit(whole_features_combined)
    a_shared = np.repeat(range(whole_features_combined.shape[0]), radius - 1)
    
    b_shared = np.fromiter(chain(*[model_shared_edges.query(whole_features_combined[v_idx], topn=radius)[1:] 
            for v_idx in range(whole_features_combined.shape[0])]), dtype=int)

    edge_latent_shared = torch.Tensor(np.stack([a_shared,b_shared])).type(torch.LongTensor)
  
    G = geomData(path_20x=torch.Tensor(features_20x),
                    path_10x=torch.Tensor(features_10x),
                    path20_edge_index=edge_spatial_20x,
                    path10_edge_index=edge_spatial_10x,
                    share_edge=edge_latent_shared) 
    return G
    


def main(data_path,radius):
    dir_path = os.path.dirname(os.path.dirname(data_path))
    save_path = os.path.join(dir_path, 'graph_files')
    file_name = Path(data_path).stem + '_radius_'+str(radius) +'.pkl'
    file_path = os.path.join(save_path, file_name)
    print('file_path:',file_path)
    os.makedirs(save_path, exist_ok=True)
    import pickle
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    for item in data:
        print('sample_name:',item['sample_name'])
        G = wsi_hypergraph(item,radius)
        item["muti_Graph"] = G

        item.pop("path_20x", None)
        item.pop("path_10x", None)

    print('Saved：',file_path)
    with open(file_path, "wb") as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type = str,required=True,
                        help='data_path')
    parser.add_argument('--radius', type=int, default=9,
                        help='number of neighboring nodes used to construct WSI hyperedges')
    args = parser.parse_args()
    results = main(args.data_path,args.radius)

