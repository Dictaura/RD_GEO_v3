import torch
import argparse

parser = argparse.ArgumentParser()


# cuda的GPU序号
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# backbone的参数
backboneParam_dict = {
    'in_size': 4,
    'out_size': 64,
    'hide_size_list': [8, 12, 16, 32],
    'n_head_list': [2, 2, 2, 2, 2],
    'n_layers': 5,
    'dropout': 0.,
    'alpha': 0.5,
    'concat': True
}

for key, value in backboneParam_dict.items():
    parser.add_argument(key, action='store_const', const=value)

backboneParam = parser.parse_args()

# critic的参数
criticParam_dict = {
    'in_size': 64,
    'out_size': 1,
    'hide_size_list': [32, 16, 8],
    'hide_size_fc': 8,
    'n_head_list': [2, 2, 2],
    'n_layers': 3,
    'graph_size': 200,
    'dropout': 0.,
    'alpha': 0.5,
    'concat': True
}

for key, value in criticParam_dict.items():
    parser.add_argument(key, action='store_const', const=value)

criticParam = parser.parse_args()

# actor的参数
actorParam_dict = {
    'in_size': 64,
    'out_size': 4,
    'hide_size_list': [32, 16, 8],
    'hide_size_fc': 8,
    'n_head_list': [2, 2, 2],
    'n_layers': 3,
    'dropout': 0.,
    'alpha': 0.5,
    'concat': True
}

for key, value in actorParam_dict.items():
    parser.add_argument(key, action='store_const', const=value)

actorParam = parser.parse_args()



