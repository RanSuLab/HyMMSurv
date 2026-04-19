from models.model_layers.layers import *
from models.model_layers.sheaf_builder import *
import numpy as np
import torch
import torch.nn as nn
from models.model_layers.cross_attention import FeedForward
from models.model_layers.fusion import AlignFusion
from models.model_layers.util import initialize_weights
import dhg
from dhg.nn import HGNNPConv
from collections import defaultdict


class HyMMSurv(nn.Module):
    def __init__(
        self,
        device,
        microbe_dim,
        k_init=8,
        path_input_dim=1536,
        n_classes=4,
        fusion="concat",
        model_size="small",
        microbe_hidden_cfg=[2048, 1024, 256],
        graph_type="HGNN",
    ):
        super(HyMMSurv, self).__init__()

        self.device = device
        self.graph_type = graph_type
        self.path_input_dim = path_input_dim
        self.n_classes = n_classes
        self.fusion = fusion
        self.microbe_dim = microbe_dim
        self.k_init = k_init

        # Path feature dims
        self.size_dict = {
            "pathomics": {
                "small": [self.path_input_dim, 256, 256],
                "large": [self.path_input_dim, 512, 256],
            }
        }

        hidden = self.size_dict["pathomics"][model_size]

        # -------- 20x encoder --------
        fc_20x = []
        for i in range(len(hidden) - 1):
            fc_20x.append(nn.Linear(hidden[i], hidden[i + 1]))
            fc_20x.append(nn.ReLU6())
            fc_20x.append(nn.Dropout(0.25))
        self.path20x_fc = nn.Sequential(*fc_20x)

        # -------- 10x encoder --------
        fc_10x = []
        for i in range(len(hidden) - 1):
            fc_10x.append(nn.Linear(hidden[i], hidden[i + 1]))
            fc_10x.append(nn.ReLU6())
            fc_10x.append(nn.Dropout(0.25))
        self.path10x_fc = nn.Sequential(*fc_10x)

        # -------- Microbe encoder --------
        micro_layers = []
        in_dim = microbe_dim
        for h in microbe_hidden_cfg:
            micro_layers.append(nn.Linear(in_dim, h))
            micro_layers.append(nn.LayerNorm(h))
            micro_layers.append(nn.GELU())
            micro_layers.append(nn.Dropout(0.25))
            in_dim = h
        self.microbe_fc = nn.Sequential(*micro_layers)

        # -------- Pathology hypergraph --------
        self.graph_p = [
            HGNNPConv(256, 256, drop_rate=0.25).to(self.device),
            HGNNPConv(256, 256, drop_rate=0.25).to(self.device),
            HGNNPConv(256, 256, use_bn=True, drop_rate=0.25).to(self.device),
        ]

        # -------- Cross-modal hypergraph --------
        self.graph_g = [
            HGNNPConv(256, 256, drop_rate=0.25).to(self.device),
            HGNNPConv(256, 256, drop_rate=0.25).to(self.device),
            HGNNPConv(256, 256, use_bn=True, drop_rate=0.25).to(self.device),
        ]

        # -------- Attention fusion --------
        self.attention_fusion = AlignFusion(embedding_dim=256, num_heads=8)

        self.feed_forward = FeedForward(256, dropout=0.25)
        self.layer_norm = nn.LayerNorm(256)

        # -------- Final fusion --------
        self.mm = nn.Sequential(
            *[nn.Linear(hidden[-1] * 2, hidden[-1] // 2), nn.ReLU6()]
        )

        self.classifier = nn.Linear(hidden[-1] // 2, self.n_classes)

        self.apply(initialize_weights)

    def forward(
        self,
        microb_feat,
        microb_flag,
        sample_name,
        muti_Graph,
        **kwargs,
    ):
        device = next(self.parameters()).device

        graph = muti_Graph.to(device)
        microb_feat = microb_feat.to(device)

        # -------- Load features --------
        x_20x = graph.path_20x
        x_10x = graph.path_10x

        feat_20x = self.path20x_fc(x_20x)
        feat_10x = self.path10x_fc(x_10x)

        n_20x = feat_20x.shape[0]
        n_10x = feat_10x.shape[0]

        pathology_features = torch.cat((feat_20x, feat_10x), dim=0)

        # -------- Build hypergraph --------
        edge_20x = self.get_hyperedge(graph.path20_edge_index)
        edge_10x = self.get_hyperedge(graph.path10_edge_index)
        edge_shared = self.get_hyperedge(graph.share_edge)

        p_hg = dhg.Hypergraph(
            num_v=pathology_features.shape[0],
            e_list=edge_20x + edge_10x + edge_shared,
        ).to(device)

        # -------- HGNN propagation --------
        for layer in self.graph_p:
            pathology_features = layer(pathology_features, p_hg)

        feat_20x_new = pathology_features[:n_20x]
        feat_10x_new = pathology_features[n_20x : n_20x + n_10x]

        # -------- Microbe embedding --------
        micro_emb = self.microbe_fc(microb_feat)

        # -------- Cross attention --------
        g_total = micro_emb.shape[0]

        attn_20x = torch.matmul(micro_emb, feat_20x_new.T)
        attn_10x = torch.matmul(micro_emb, feat_10x_new.T)

        edge_cross_20x = self.build_cross_hyperedge(
            attn_20x, g_total, feat_20x_new, self.k_init
        )

        edge_cross_10x = self.build_cross_hyperedge(
            attn_10x, g_total, feat_10x_new, self.k_init
        )

        edge_cross_10x = self.offset_hyperedges(
            edge_cross_10x, offset=n_20x
        )

        cross_edges = edge_cross_20x + edge_cross_10x

        token = torch.cat((micro_emb, pathology_features), dim=0)

        if cross_edges is not None:
            g_hg = dhg.Hypergraph(
                num_v=token.shape[0], e_list=cross_edges
            ).to(device)

            for layer in self.graph_g:
                token = layer(token, g_hg)

        # -------- Attention fusion --------
        token = token.unsqueeze(0)
        token, attn_path, attn_microb = self.attention_fusion(token)

        token = self.feed_forward(token)
        token = self.layer_norm(token)

        gene_embed = torch.mean(token[:, :g_total, :], dim=1)
        path_embed = torch.mean(token[:, g_total:, :], dim=1)

        fusion = self.mm(torch.cat([gene_embed, path_embed], dim=1))

        logits = self.classifier(fusion)
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)

        out = {
            "fused": fusion,
            "attn_20x": attn_20x.squeeze(),
            "attn_10x": attn_10x.squeeze(),
            "attn_path": attn_path.squeeze(),
            "attn_microb": attn_microb.squeeze(),
        }

        return hazards, out, S

    def get_hyperedge(self, edge):
        # Convert pair-wise edges to hyperedges
        adj = edge.cpu().numpy()
        hyperedges = defaultdict(set)

        for s, t in adj.T:
            if s != t:
                hyperedges[s].add(t)

        e_list = []
        for s, nbrs in hyperedges.items():
            he = {s}.union(nbrs)
            if len(he) >= 2:
                e_list.append(list(he))

        return e_list

    def build_cross_hyperedge(
        self,
        attn_scores,
        g_total,
        pathology_features,
        k_init=8,
    ):
        # Build cross-modal hyperedges via top-k attention
        n_path = pathology_features.shape[0]
        k = min(k_init, n_path)

        _, top_idx = torch.topk(attn_scores, k, dim=1)

        source = torch.arange(g_total, device=self.device).repeat_interleave(k)
        target = top_idx.flatten() + g_total

        e_list = self.get_hyperedge(
            torch.stack([source.to(self.device), target.to(self.device)])
        )

        return e_list

    def offset_hyperedges(self, e_list, offset, gene_id=0):
        # Shift pathology node indices
        new_e_list = []

        for e in e_list:
            new_e = []
            for v in e:
                if v == gene_id:
                    new_e.append(gene_id)
                else:
                    new_e.append(v + offset)
            new_e_list.append(new_e)

        return new_e_list