# %%
import os
import sys
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader, DataListLoader, Batch, Data

sys.path.append('/Users/xichen/Documents/paper2-traj-pred/carla-data/VectorNet')

# %%
from model.layers.global_graph import GlobalGraph, SelfAttentionFCLayer
from model.layers.subgraph import SubGraph
from model.layers.basic_module import MLP
from model.backbone.vectornet_v2 import VectorNetBackbone
# %%
from utils.loss import VectorLoss
from dataloader.carla_scene_loader import GraphData, CarlaInMem
# %%
class VectorNet(nn.Module):
    """
    hierarchical GNN with trajectory prediction MLP
    """

    def __init__(self,
                 in_channels=8,
                 horizon=50,
                 num_subgraph_layers=3,
                 num_global_graph_layer=1,
                 subgraph_width=64,
                 global_graph_width=64,
                 traj_pred_mlp_width=64,
                 with_aux:bool=False,
                 device=torch.device('cpu')):
        super(VectorNet, self).__init__()
        #some params
        self.polyline_vec_shape = in_channels * (2 ** num_subgraph_layers)
        self.out_channels = 2
        self.horizon = horizon
        self.subgraph_width = subgraph_width
        self.global_graph_width = global_graph_width
        self.k = 1

        self.device = device

        # subgraph feature extractor
        self.backbone = VectorNetBackbone(
            in_channels=in_channels,
            num_subgraph_layers=num_subgraph_layers,
            subgraph_width=subgraph_width,
            num_global_graph_layer=num_global_graph_layer,
            global_graph_width=global_graph_width,
            with_aux=with_aux,
            device=device
        )

        # pred mlp
        self.traj_pred_mlp = nn.Sequential(
            MLP(global_graph_width, traj_pred_mlp_width, traj_pred_mlp_width),
            nn.Linear(traj_pred_mlp_width, self.horizon * self.out_channels)
        )

    def forward(self, data):
        """
        args:
            data (Data): [x, y, cluster, edge_index, valid_len]
        """
        global_feat, aux_out, aux_gt = self.backbone(data)              # [batch_size, time_step_len, global_graph_width]
        target_feat = global_feat[:, 0]

        pred = self.traj_pred_mlp(target_feat)

        return {"pred": pred, "aux_out": aux_out, "aux_gt":aux_gt}

    def inference(self, data):
        batch_size = data.num_graphs

        pred_traj = self.forward(data)["pred"].view((batch_size, self.k, self.horizon, 2)).cumsum(2)

        return pred_traj
    
# %%
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2

    in_channels, pred_len = 10, 30
    show_every = 10
    # os.chdir('..')
    model = VectorNet(in_channels, pred_len, with_aux=True).to(device)
    INTERMEDIATE_DATA_DIR = "../../scene_mining_intermediate"
    dataset_input_path = os.path.join(os.path.dirname(__file__), INTERMEDIATE_DATA_DIR)
    dataset = CarlaInMem(dataset_input_path)
    data_iter = DataLoader(dataset, batch_size=batch_size, num_workers=1, pin_memory=True)

    #train_mode
    model.train()
    for i, data in enumerate(data_iter):
        # out, aux_out, mask_feat_gt = model(data)
        loss = model.loss(data.to(device))
        print("Trainng Pass! loss: {}".format(loss))

        if i == 2:
            break
    
    # eval mode
    model.eval()
    for i, data in enumerate(data_iter):
        out = model(data.to(device))
        print("Evaluation Pass! Shape of out: {]".format(out.shape))

        if i == 2:
            break

