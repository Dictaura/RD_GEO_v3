import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric.nn.conv as conv_g

class GAT_Multi_heads(nn.Module):
    def __init__(self, in_size, out_size, hide_size, n_heads):
        super(GAT_Multi_heads, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.hide_size = hide_size
        self.n_heads = n_heads

        self.GAT = conv_g.GATConv(self.in_size, self.hide_size, self.n_heads, bias=False)
        self.GCN = conv_g.GCNConv(self.hide_size * self.n_heads, self.out_size, bias=False, normalize=True)

    def forward(self, x, edge_index, edge_weight=None):
        y_gat = self.GAT(x, edge_index)
        y = self.GCN(y_gat, edge_index, edge_weight=edge_weight)
        return y


class BackboneNet(nn.Module):
    def __init__(self, in_size, out_size, hide_size_list, n_head_list, n_layers, concat=True):
        super(BackboneNet, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.hide_size_list = hide_size_list
        self.n_head_list = n_head_list
        self.n_layers = n_layers
        self.concat = concat
        self.size_layer_list = [in_size] + hide_size_list + [out_size]

        self.layers_gat = []

        for i in range(n_layers):
            # self.layers_gat.append(GAT_Multi_heads(self.size_layer_list[i],
            #                                        self.size_layer_list[i+1],
            #                                        self.size_layer_list[i+1],
            #                                        self.n_head_list[i]))
            self.layers_gat.append(conv_g.GATConv(self.size_layer_list[i], self.size_layer_list[i + 1]))
            self.add_module('GAT_block_{}'.format(i), self.layers_gat[i])

        self.layers_gat = nn.ModuleList(self.layers_gat)

    def forward(self, x, edge_index, edge_weight=None):
        for layer in self.layers_gat:
            # x = layer(x, edge_index, edge_weight=edge_weight)
            x = layer(x, edge_index)
            x = F.relu(x)
        return x


class ActorNet(nn.Module):
    def __init__(self, in_size, out_size, hide_size_list, hide_size_fc, n_head_list, n_layers, concat=True):
        super(ActorNet, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.hide_size_list = hide_size_list
        self.hide_size_fc = hide_size_fc
        self.n_head_list = n_head_list
        self.n_layers = n_layers
        self.concat = concat
        self.size_layer_list = [in_size] + hide_size_list

        self.layers_gat = []

        for i in range(self.n_layers):
            # self.layers_gat.append(GAT_Multi_heads(self.size_layer_list[i],
            #                                        self.size_layer_list[i + 1],
            #                                        self.size_layer_list[i + 1],
            #                                        self.n_head_list[i]))
            self.layers_gat.append(conv_g.GATConv(self.size_layer_list[i], self.size_layer_list[i + 1]))
            self.add_module('GAT_block_{}'.format(i), self.layers_gat[i])

        self.layers_gat = nn.ModuleList(self.layers_gat)

        self.fc1 = nn.Linear(self.size_layer_list[-1], self.hide_size_fc, bias=False)
        self.norm1 = nn.BatchNorm1d(self.hide_size_fc)
        self.fc2 = nn.Linear(self.hide_size_fc, self.out_size, bias=False)
        self.norm2 = nn.BatchNorm1d(self.out_size)

    def forward(self, x, edge_index, max_size,edge_weight=None):
        for layer in self.layers_gat:
            # x = layer(x, edge_index, edge_weight)
            x = layer(x, edge_index)
            x = F.relu(x)

        x = F.relu(self.fc1(x))
        x = self.norm1(x)
        x = F.relu(self.fc2(x))
        x = self.norm2(x)

        x = x.view(x.size(0)//max_size, max_size, -1)
        x = torch.flatten(x, 1, 2)
        action_prob = F.softmax(x, dim=1)

        return action_prob


class CriticNet(nn.Module):
    def __init__(self, in_size, out_size, hide_size_list, hide_size_fc, n_head_list, n_layers, concat=True):
        super(CriticNet, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.hide_size_list = hide_size_list
        self.hide_size_fc = hide_size_fc
        self.n_head_list = n_head_list
        self.n_layers = n_layers
        self.concat = concat
        self.size_layer_list = [in_size] + hide_size_list

        self.layers_gat = []

        for i in range(self.n_layers):
            # self.layers_gat.append(GAT_Multi_heads(self.size_layer_list[i],
            #                                        self.size_layer_list[i + 1],
            #                                        self.size_layer_list[i + 1],
            #                                        self.n_head_list[i]))
            self.layers_gat.append(conv_g.GATConv(self.size_layer_list[i], self.size_layer_list[i + 1]))
            self.add_module('GAT_block_{}'.format(i), self.layers_gat[i])

        self.layers_gat = nn.ModuleList(self.layers_gat)

        self.fc1 = nn.Linear(self.size_layer_list[-1], self.hide_size_fc, bias=False)
        self.norm1 = nn.BatchNorm1d(self.hide_size_fc)
        self.fc2 = nn.Linear(self.hide_size_fc, self.out_size, bias=False)
        self.norm2 = nn.BatchNorm1d(self.out_size)

    def forward(self, x, edge_index, max_size,edge_weight=None):
        for layer in self.layers_gat:
            # x = layer(x, edge_index, edge_weight)
            x = layer(x, edge_index)
            x = F.relu(x)

        x = F.relu(self.fc1(x))
        x = self.norm1(x)
        x = F.relu(self.fc2(x))
        x = self.norm2(x)

        x = x.view(x.size(0)//max_size, max_size, -1)
        value = torch.sum(x, dim=1)

        return value