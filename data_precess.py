import os
import torch
from data.process import generate_rfam_all
from utils.rna_lib import graph_padding


root = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
    data_dir = root + '/data/raw/rfam_learn/train/'
    save_dir = root + '/data/processed/rfam_learn/train/all.pt'
    generate_rfam_all(data_dir, save_dir)
    # data_dir = root + '/data/processed/rfam_learn/test/all.pt'
    # graph_list = torch.load(data_dir)
    # for graph in graph_list:
    #     print(graph.y['seq_base'])
    # graph_list = torch.load(data_dir)
    # graph = graph_list[0]
    # max_size = 450
    # graph = graph_padding(graph, max_size)
    # print(1)