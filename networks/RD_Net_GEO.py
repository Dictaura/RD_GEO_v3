import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric.nn.conv as conv_g

class GAT_Multi_heads(nn.Module):
    def __init__(self, in_size, out_size, hide_size, n_heads, use_linear=False):
        super(GAT_Multi_heads, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.hide_size = hide_size
        self.n_heads = n_heads

        self.GAT = conv_g.GATConv(self.in_size, self.hide_size, self.n_heads, bias=False)
        self.fc = None
        if use_linear:
            self.fc = nn.Linear(self.hide_size * self.n_heads, self.out_size, bias=False)

    def forward(self, x, edge_index, edge_weight=None):
        y_gat = self.GAT(x, edge_index)
        y = y_gat
        if self.fc is not None:
            y = F.relu(self.fc(y_gat))
        return y


class BackboneNet(nn.Module):
    def __init__(self, in_size, out_size, hide_size_list, n_head_list, n_layers, conv1d_size, concat=True, use_conv1d=False, use_linear=False):
        super(BackboneNet, self).__init__()
        self.in_size = in_size
        self.conv1d_size = conv1d_size
        self.out_size = out_size
        self.hide_size_list = hide_size_list
        self.n_head_list = n_head_list
        self.n_layers = n_layers
        self.concat = concat
        self.size_layer_list = [in_size] + hide_size_list + [out_size]

        self.conv1d = None

        if use_conv1d:
            self.size_layer_list = [conv1d_size] + self.size_layer_list
            self.conv1d = nn.Conv1d(in_size, conv1d_size, kernel_size=7, stride=1, padding=3)

        self.layers_gat = []

        for i in range(n_layers):
            self.layers_gat.append(GAT_Multi_heads(self.size_layer_list[i],
                                                   self.size_layer_list[i+1],
                                                   self.size_layer_list[i+1],
                                                   self.n_head_list[i]))
            self.add_module('GAT_block_{}'.format(i), self.layers_gat[i])

        self.layers_gat = nn.ModuleList(self.layers_gat)

    def forward(self, x, edge_index, max_size=100, edge_weight=None):
        if self.conv1d is not None:
            x1 = x.view(-1, max_size, self.in_size)
            x1 = x1.permute(0, 2, 1)
            x1 = F.relu(self.conv1d(x1))
            x1 = x1.permute(0, 2, 1)
            x1 = torch.flatten(x1, 0, 1)
            x = torch.cat([x, x1], dim=1)
        for layer in self.layers_gat:
            x = layer(x, edge_index, edge_weight=edge_weight)
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
            self.layers_gat.append(GAT_Multi_heads(self.size_layer_list[i],
                                                   self.size_layer_list[i + 1],
                                                   self.size_layer_list[i + 1],
                                                   self.n_head_list[i]))
            self.add_module('GAT_block_{}'.format(i), self.layers_gat[i])

        self.layers_gat = nn.ModuleList(self.layers_gat)

        self.fc1 = nn.Linear(self.size_layer_list[-1], self.hide_size_fc, bias=False)
        self.fc2 = nn.Linear(self.hide_size_fc, self.out_size, bias=False)

    def forward(self, x, edge_index, max_size,edge_weight=None):
        for layer in self.layers_gat:
            x = layer(x, edge_index, edge_weight)
            x = F.relu(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

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
            self.layers_gat.append(GAT_Multi_heads(self.size_layer_list[i],
                                                   self.size_layer_list[i + 1],
                                                   self.size_layer_list[i + 1],
                                                   self.n_head_list[i]))
            self.add_module('GAT_block_{}'.format(i), self.layers_gat[i])

        self.layers_gat = nn.ModuleList(self.layers_gat)

        self.fc1 = nn.Linear(self.size_layer_list[-1], self.hide_size_fc, bias=False)
        self.fc2 = nn.Linear(self.hide_size_fc, self.out_size, bias=False)

    def forward(self, x, edge_index, max_size,edge_weight=None):
        for layer in self.layers_gat:
            x = layer(x, edge_index, edge_weight)
            x = F.relu(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = x.view(x.size(0)//max_size, max_size, -1)
        value = torch.sum(x, dim=1)

        return value