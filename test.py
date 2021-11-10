import os

import Levenshtein
import RNA
import numpy as np
import random
from random import shuffle, sample


import torch

from utils.rna_lib import edge_distance, get_edge_h

root = os.path.dirname(os.path.realpath(__file__))

data_dir = root + '/data/processed/rfam_learn/train/all.pt'
dataset = torch.load(data_dir)
dataset = sample(dataset, 10)

print(1)

# real_edge = get_edge_h('.........\n')
#
# aim_edge = get_edge_h('.........')
#
# distance = edge_distance(real_edge, aim_edge)
#
# distance = Levenshtein.distance('.........\n', '.........')
#
# print(distance)

loop_index = RNA.make_loop_index('.....((((((((...((((((((((..((((..........))))..((((..........))))..((((..........))))..))))))))))...((((((((((..((((..........))))..((((..........))))..((((..........))))..))))))))))...((((((((((..((((..........))))..((((..........))))..((((..........))))..))))))))))...)))))))).....')

print(loop_index)

