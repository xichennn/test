# source: https://github.com/xk-huang/yet-another-vectornet
import numpy as np
import torch
import torch.nn as nn

from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, max_pool, avg_pool
from torch_geometric.utils import add_self_loops, remove_self_loops

from model.layers.basic_module import MLP

class SubGraph1(nn.Module):
    """
    Subgraph that computes all vectors in a polyline, and get a polyline-level feature
    """

    def __init__(self, in_channels, num_subgraph_layers=3, hidden_unit=64):
        super(SubGraph1, self).__init__()
        self.num_subgraph_layers = num_subgraph_layers
        self.out_channels = hidden_unit * 2

        self.layer_seq = nn.Sequential()
        for i in range(num_subgraph_layers):
            self.layer_seq.add_module(
                f'glp_{i}', GraphLayerProp(in_channels, hidden_unit)
            )
            in_channels = hidden_unit * 2


    def forward(self, sub_data):
        """
        polyline vector set in torch_geometric.data.Data format
        param:
        sub_data(Data): [x, y, cluster, edge_index, valid_len]
        """

        x, edge_index, batch = sub_data.x, sub_data.edge_index, sub_data.batch

        for name, layer in self.layer_seq.named_modules():
            if isinstance(layer, GraphLayerProp):
                x = layer(x, edge_index)
        sub_data.x = x
        out_data = max_pool(sub_data.cluster, sub_data)

        assert out_data.x.shape[0] % int(sub_data.time_step_len[0]) == 0
        out_data.x = out_data.x / (out_data.x.norm(dim=0) + 1e-12) #L2 normalization
        return out_data
    
class GraphLayerProp(MessagePassing):
    """
    Message Passing mechanism for information aggregation
    """

    def __init__(self, in_channels, hidden_unit=64, verbose=False):
        super(GraphLayerProp, self).__init__(
            aggr='max') #MaxPooling aggregation
        self.verbose = verbose
        self.residual = True if in_channels == hidden_unit else False

        self.mlp = MLP(in_channels, hidden_unit, hidden_unit)

    def forward(self, x, edge_index):
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        if self.verbose:
            print(f'x before mlp: {x}')

        x = self.mlp(x)

        if self.verbose:
            print(f'x after mlp: {x}')
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)
    
    def message(self, x_j):
        return x_j
    
    def update(self, aggr_out, x):
        if self.verbose:
            print(f"x after mlp: {x}")
            print(f"aggr_out: {aggr_out}")
        return torch.cat([x, aggr_out], dim=1)


class SubGraph2(nn.Module):
    """
    Subgraph that computes all vectors in a polyline, and get a polyline-level feature
    """

    def __init__(self, in_channels, num_subgraph_layers=3, hidden_unit=64):
        super(SubGraph2, self).__init__()
        self.num_subgraph_layers = num_subgraph_layers
        self.out_channels = hidden_unit * 2

        self.layer_seq = nn.Sequential()
        for i in range(num_subgraph_layers):
            self.layer_seq.add_module(
                f'glp_{i}', MLP(in_channels, hidden_unit)
            )
            in_channels = hidden_unit * 2


    def forward(self, sub_data, category):
        """
        polyline vector set in torch_geometric.data.Data format
        param:
        sub_data(Data): [x, y, cluster, edge_index, valid_len]
        """

        x = sub_data.x
        category = category.long()

        for _, layer in self.layer_seq.named_modules():
            if isinstance(layer, MLP):

                x = layer(x)
                sub_data.x = x
                agg_data = max_pool(category, sub_data)

                x = torch.cat([x, agg_data.x[category]], dim=-1)
        sub_data.x = x
        # out_data = max_pool(category, sub_data)

        # assert sub_data.x.shape[0] % int(sub_data.time_step_len[0]) == 0
        sub_data.x = sub_data.x / (sub_data.x.norm(dim=0) + 1e-12) #L2 normalization
        return sub_data.x
    
if __name__ == "__main__":
    data = Data(x=torch.tensor([[1.0], [7.0]]), edge_index=torch.tensor([[0, 1], [1, 0]]))
    print(data)
    layer = GraphLayerProp(1, 1, True)
    for k, v in layer.state_dict().items():
        if k.endswith('weight'):
            v[:] = torch.tensor([[1.0]])
        elif k.endswith('bias'):
            v[:] = torch.tensor([1.0])
    y = layer(data.x, data.edge_index)
