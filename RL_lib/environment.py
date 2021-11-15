import gym
import torch
from tqdm import tqdm
from torch import nn
from torch.nn import functional as F
import numpy as np
import os
from torch_geometric.data import DataLoader as DataLoader_g
from utils.rna_lib import random_init_sequence, random_init_sequence_pair, graph_padding, forbidden_actions_pair, \
    get_distance_from_graph_norm, get_edge_h, get_topology_distance, rna_act_pair, get_energy_from_graph, \
    get_distance_from_graph, get_topology_distance_norm
from collections import namedtuple
import torch_geometric
from utils.config_ppo import device
import multiprocessing as mp
from functools import partial
import pathos.multiprocessing as pathos_mp


Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])


class RNA_Graphs_Env(gym.Env):
    """
    多RNA图的环境
    """
    def __init__(self, dataset, cal_freq=1, max_size=None, pool=None,
                 reward_type='energy', done_type='distance', distance_type='hamming', init='unpair', action_space=4):
        """
        多RNA环境初始化
        :param dataset: 数据集，为graph的batch
        :param batch_size:
        """
        super(RNA_Graphs_Env, self).__init__()
        # self.graphs = []
        # 加载数据
        # dataloader = DataLoader_g(dataset, len(dataset), shuffle=True)
        # for graph in dataloader:
        #     self.graphs = graph
        # 大图转列表
        # pool = torch.multiprocessing.Pool()
        if pool is None:
            self.pool = pathos_mp.ProcessingPool()
        else:
            self.pool = pool
        if max_size is not None:
            partial_work = partial(graph_padding, max_size=max_size)
            graphs_ = self.pool.map(partial_work, dataset)

            self.graphs_ = list(graphs_)
        else:
            self.graphs_ = dataset
        self.graphs = torch_geometric.data.Batch.from_data_list(self.graphs_).clone()
        # self.graphs_ = self.graphs.clone().to_data_list()
        self.graphs = self.graphs.to_data_list()
        self.last_energy_list = np.zeros(len(self.graphs))
        self.last_distance_list = np.zeros(len(self.graphs))
        # self.last_ratio_list = np.zeros(len(self.graphs))
        self.len_list = [len(graph.y['dotB']) for graph in self.graphs]
        self.ids = list(range(len(self.graphs)))
        self.forbidden_actions_list = [[]] * len(self.graphs)
        self.cal_freq = cal_freq
        self.aim_edge_h_list = []
        self.reward_type = reward_type
        self.done_type = done_type
        self.distance_type = distance_type
        self.init = init
        self.action_space = action_space
        pass

    def reset(self):
        """
        环境的复位函数
        :return: 复位的图
        """
        self.graphs = torch_geometric.data.Batch.from_data_list(self.graphs_).clone().to_data_list()
        self.last_energy_list = np.zeros(len(self.graphs))
        self.last_distance_list = np.zeros(len(self.graphs))
        # self.last_ratio_list = np.zeros(len(self.graphs))
        self.len_list = [len(graph.y['dotB']) for graph in self.graphs]
        self.ids = list(range(len(self.graphs)))
        dotB_list = [graph.y['dotB'] for graph in self.graphs]
        edge_index_list = [graph.edge_index for graph in self.graphs]
        self.aim_edge_h_list = list(self.pool.map(get_edge_h, dotB_list))

        max_size = self.graphs[0].x.shape[0]
        if self.init == 'unpair':
            init_work = partial(random_init_sequence, max_size=max_size)
            init_result = self.pool.map(init_work, dotB_list)
        else:
            init_work = partial(random_init_sequence_pair, max_size=max_size, action_space=self.action_space)
            init_result = self.pool.map(init_work, dotB_list, edge_index_list)

        init_result = list(init_result)
        init_result = list(zip(*init_result))
        for i in range(len(self.graphs)):
            self.graphs[i].y['seq_base'], self.graphs[i].x = init_result[0][i], init_result[1][i]

        self.last_energy_list = self.pool.map(get_energy_from_graph, self.graphs)
        self.last_energy_list = np.array(list(self.last_energy_list))

        if self.distance_type == 'hamming':
            self.last_distance_list = self.pool.map(get_distance_from_graph, self.graphs)
        elif self.distance_type == 'hamming_norm':
            self.last_energy_list = self.pool.map(get_distance_from_graph_norm, self.graphs)
        elif self.distance_type == 'topo':
            self.last_distance_list = self.pool.map(get_topology_distance, self.graphs, self.aim_edge_h_list)
        elif self.distance_type == 'topo_norm':
            self.last_distance_list = self.pool.map(get_topology_distance_norm, self.graphs, self.aim_edge_h_list)
        # self.last_distance_list = self.pool.map(get_distance_from_graph_norm, self.graphs)
        self.last_distance_list = np.array(list(self.last_distance_list))

        forbid_work = partial(forbidden_actions_pair, action_space=self.action_space)
        # test_list = forbid_work(self.graphs[0])
        self.forbidden_actions_list = list(self.pool.map(forbid_work, self.graphs))

        return torch_geometric.data.Batch.from_data_list(self.graphs).clone().to_data_list()

    def reset_for_test(self):
        """
        环境的复位函数
        :return: 复位的图
        """
        self.last_energy_list = np.zeros(len(self.graphs))
        self.last_distance_list = np.zeros(len(self.graphs))
        # self.last_ratio_list = np.zeros(len(self.graphs))
        self.len_list = [len(graph.y['dotB']) for graph in self.graphs]
        for i in range(len(self.graphs)):
            self.graphs[i].y['seq_base'], self.graphs[i].x = random_init_sequence(self.graphs[i].y['dotB'], self.graphs[i].x.shape[0])
            self.last_energy_list[i] = get_energy_from_graph(self.graphs[i])
            self.last_distance_list[i] = get_distance_from_graph_norm(self.graphs[i])
            # self.last_ratio_list[i] = get_pair_ratio(self.graphs[i])
            self.forbidden_actions_list[i] = forbidden_actions_pair(self.graphs[i])
        return torch_geometric.data.Batch.from_data_list(self.graphs).clone().to_data_list()

    def step(self, actions, ep):
        """
        环境接受并执行动作
        :param actions: 动作编号
        :param reward_type: reward
        :return:
        """
        work = partial(rna_act_pair, action_space=self.action_space)
        result = self.pool.map(work, actions.cpu(), self.graphs, self.forbidden_actions_list)

        results = list(result)
        results = list(zip(*results))
        self.graphs = list(results[0])
        self.forbidden_actions_list = list(results[1])
        if ep % self.cal_freq == 0:
            energy_list = self.pool.map(get_energy_from_graph, self.graphs)
            energy_list = np.array(list(energy_list))

            if self.distance_type == 'hamming':
                distance_list = self.pool.map(get_distance_from_graph, self.graphs)
            elif self.distance_type == 'hamming_norm':
                distance_list = self.pool.map(get_distance_from_graph_norm, self.graphs)
            elif self.distance_type == 'topo':
                distance_list = self.pool.map(get_topology_distance, self.graphs, self.aim_edge_h_list)
            elif self.distance_type == 'topo_norm':
                distance_list = self.pool.map(get_topology_distance_norm, self.graphs, self.aim_edge_h_list)

            distance_list = np.array(list(distance_list))
            # ratio_list = np.array(list(map(get_pair_ratio, self.graphs)))

        else:
            energy_list = self.last_energy_list
            distance_list = self.last_distance_list
            # ratio_list = self.last_ratio_list

        # 根据reward_type计算reward
        if self.reward_type is 'ratio':
            pass
            # reward_list = ratio_list #- self.last_ratio_list
        elif self.reward_type is 'energy':
            reward_list = self.last_energy_list - energy_list
        elif self.reward_type is 'distance':
            reward_list = self.last_distance_list - distance_list
            # reward_list = [(1 - distance / length) for distance, length in zip(distance_list, self.len_list)]
            # reward_list = 1 - distance_list
        # 根据done_type判断是否完成
        is_terminal = 0
        # if done_type is 'ratio':
        #     if np.all(np.array(ratio_list) == 1.):
        #         is_terminal = 1
        #     done_list = np.where(np.array(ratio_list) == 1., 10, 1)
        if self.done_type is 'distance':
            if np.all(np.array(distance_list) == 0):
                is_terminal = 1
            done_list = np.where(np.array(distance_list) == 0, 10, 1)
        reward_list = np.array(reward_list) * done_list

        # 更新环境信息
        self.last_energy_list = energy_list
        self.last_distance_list = distance_list
        # self.last_ratio_list = ratio_list

        return torch_geometric.data.Batch.from_data_list(self.graphs).clone().to_data_list(), reward_list, is_terminal, done_list, self.ids.copy()

    def remove_graph(self, orders):
        """
        将设计完的图暂时移除
        :param orders: 设计完成的图当前的序号
        :return: 完成的图、完成的图的id、完成的图的点括号、完成的图的序列、完成图的自由能、完成图的目标距离
        """
        # 记录完成图的信息
        remove_id = [self.ids[i] for i in range(0, len(self.ids), 1) if i in orders]
        sequence = [self.graphs[i].y['seq_base'] for i in range(0, len(self.graphs), 1) if i in orders]
        dotB = [self.graphs[i].y['dotB'] for i in range(0, len(self.graphs), 1) if i in orders]
        distance = [self.last_distance_list[i] for i in range(0, len(self.last_distance_list), 1) if i in orders]
        energy = [self.last_energy_list[i] for i in range(0, len(self.last_energy_list), 1) if i in orders]
        graph = [self.graphs[i] for i in range(0, len(self.graphs), 1) if i in orders]
        # 从暂存区删除完成图的记录
        self.graphs = [self.graphs[i] for i in range(0, len(self.graphs), 1) if i not in orders]
        self.ids = np.delete(np.array(self.ids), orders).tolist()
        self.last_energy_list = np.delete(self.last_energy_list, orders)
        self.last_distance_list = np.delete(self.last_distance_list, orders)
        # self.last_ratio_list = np.delete(self.last_ratio_list, orders)
        self.len_list = np.delete(np.array(self.len_list), orders).tolist()
        forbidden_actions_list = []
        for i in range(0, len(self.graphs), 1):
            if i not in orders:
                forbidden_actions_list.append(self.forbidden_actions_list[i])
        self.forbidden_actions_list = forbidden_actions_list
        self.aim_edge_h_list = [self.aim_edge_h_list[i] for i in range(0, len(self.graphs), 1) if i not in orders]
        # print('Graph_{} is removed!'.format(remove_id))
        # print('Struct: {} | Sequence: {} | Energy: {} | Distance: {}'.format(dotB, sequence, energy, distance))

        return graph, remove_id, dotB, sequence, energy, distance








