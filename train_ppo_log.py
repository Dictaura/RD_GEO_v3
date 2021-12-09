import math
from RL_lib.environment import RNA_Graphs_Env
from networks.RD_Net_GEO import BackboneNet, CriticNet, ActorNet
from torch.autograd import Variable
from utils.config_ppo import device, backboneParam, criticParam, actorParam, num_change
import os
import torch
import torch_geometric
from RL_lib.ppo_log import PPO_Log, Transition
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from utils.rna_lib import seq_onehot2Base, get_energy_from_onehot, get_energy_from_base, \
    get_distance_from_base, get_topology_distance, get_pair_ratio
from random import choice, sample
import multiprocessing as mp
import pathos.multiprocessing as pathos_mp
import cProfile
import re
import seaborn as sns
import matplotlib.pyplot as plt


def main():

    ################################### Training ###################################

    print("===============================train===============================")

    ####### initialize global hyperparameters ######

    torch.backends.cudnn.enable = False

    # torch.multiprocessing.set_start_method('spawn')

    # 根目录
    root = os.path.dirname(os.path.realpath(__file__))

    # 当前时间
    local_time = str(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))

    # 设置进程池
    pool_main = pathos_mp.ProcessingPool()
    pool_agent = pathos_mp.ProcessingPool()
    pool_env = pathos_mp.ProcessingPool()

    #####################################################

    ###################### logging ######################

    # 目录
    # 记录总目录
    log_dir_root = root + "/logs/PPO_logs_" + local_time
    if not os.path.exists(log_dir_root):
        os.makedirs(log_dir_root)

    # 时间统计
    # 程序时间统计
    time_sum_dir = log_dir_root + '/program.prof'
    cProfile.run('re.compile("ccc")', filename=time_sum_dir)

    # 日志位置
    log_dir = log_dir_root + '/Logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 模型保存目录
    model_dir = log_dir_root + '/Model/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # tensorboard的目录
    tensor_dir = log_dir_root + '/Tensorboard/'
    if not os.path.exists(tensor_dir):
        os.makedirs(tensor_dir)

    # create new log file for each run
    log_f_name = log_dir + "/log"
    done_log_f_name = log_dir + "/done_log.csv"
    print("logging at : " + log_f_name)

    # create tensorboard log as writer
    writer = SummaryWriter(tensor_dir, comment="Train_Log.log")
    writer.add_text('file', str(local_time))
    print('tensorboard at:' + tensor_dir)

    #####################################################

    ####### initialize environment hyperparameters ######

    url_data = root + '/data/processed/rfam_learn/train/all.pt'
    dataset = torch.load(url_data)
    # dataset = sample(dataset, 10)
    dataset = [dataset[0]]
    len_list = [len(graph.y['dotB']) for graph in dataset]
    max_size = max(len_list)

    # 总步数
    max_train_timestep = int(60000)

    # 每episode的步数
    max_ep_len = 200

    # 打印训练结果间隔步数
    print_freq = 4 * max_ep_len

    # 日志记录间隔
    log_freq = 1 * max_ep_len

    # 模型保存间隔
    save_model_freq = 20 * max_ep_len

    # 动作选择模式
    action_type = 'selectAction'

    # 选择多步计算频率
    cal_freq_start = 1 # max_ep_len
    cal_freq_end = 1 # max_ep_len
    cal_freq_decay = 20000

    # reward结算模式和done判定模式
    reward_type = 'distance'
    done_type = 'distance'
    distance_type = 'hamming'
    init = 'pair'
    action_space = num_change

    env = RNA_Graphs_Env(dataset, cal_freq=cal_freq_start, max_size=max_size, pool=pool_env,
                         reward_type=reward_type, done_type=done_type, distance_type=distance_type,
                         action_space=action_space, init=init)

    #####################################################

    ################ PPO hyperparameters ################

    # backbone、ctritc、actor的参数详见 ./utilities/config
    # 模型训练间隔
    update_timestep = max_ep_len

    # 每次训练的epoch
    K_epochs = 6

    batch_size = 128

    # ppo的更新限制
    eps_clip = 0.1

    # actor参数冻结轮次
    actor_freeze_ep = 0

    # 学习率衰减率
    lr_decay = 0.999

    # 奖励衰减
    gamma = 0.9

    # 学习率
    lr_backbone = 0.000001
    lr_actor = 0.0000001  # learning rate for actor network
    lr_critic = 0.0000001  # learning rate for critic network

    agent = PPO_Log(BackboneNet, ActorNet, CriticNet,
        backboneParam, criticParam, actorParam,
                    lr_backbone=lr_backbone, lr_critic=lr_critic, lr_actor=lr_actor, train_batch_size=batch_size,
                K_epoch=K_epochs, eps_clips=eps_clip, actor_freeze_ep=actor_freeze_ep, num_graph=len(env.len_list), gamma=gamma,
                pool=pool_agent, action_space=action_space).to(device)

    # 加载模型
    # agent.load(root + '/logs/PPO_logs_2021_11_16_16_44_28/Model/', 100)

    # 学习率更新策略
    scheduler_b = torch.optim.lr_scheduler.ExponentialLR(agent.optimizer_b, lr_decay)
    scheduler_c = torch.optim.lr_scheduler.ExponentialLR(agent.optimizer_c, lr_decay)
    scheduler_a = torch.optim.lr_scheduler.ExponentialLR(agent.optimizer_a, lr_decay)

    #####################################################

    ################# hyperparameters log ################

    writer.add_text("loop_step", str(max_ep_len))
    writer.add_text("reward_type", reward_type)
    writer.add_text("done_type", done_type)
    writer.add_text("init", init)
    writer.add_text("distance type", distance_type)
    writer.add_text("lr_actor", str(lr_actor))
    writer.add_text("lr_critic", str(lr_critic))
    writer.add_text("batch_size", str(batch_size))

    #####################################################

    ################# training procedure ################

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("============================================================================================")

    # logging file
    log_f_list = []
    act_log_f_list = []
    place_log_f_list = []
    for i in range(len(env.graphs_)):
        log_f_list.append(open(log_f_name + '_' + str(i) + '.csv', "w+"))
        act_log_f_list.append(open(log_f_name + '_act_' + str(i) + '.csv', "w+"))
        # place_log_f_list.append(open(log_f_name + '_place_' + str(i) + '.csv', "w+"))
    done_log_f = open(done_log_f_name, "w+")
    for log_f in log_f_list:
        log_f.write('episode, timestep, reward, energy, distance, dotB, sequence' + '\n')

    # 初始化变量
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    min_list = []
    reward_list = []

    # 游戏循环
    while time_step < max_train_timestep:
        i_episode += 1

        print("=====================================" + str(i_episode) + "==================================================")
        # 重置reward记录
        current_ep_reward = 0
        # current_ep_reward_np = torch.tensor(np.zeros(len(env.len_list)), dtype=float)

        # 重置环境，获得初始状态
        state = env.reset()
        # ratio = get_pair_ratio(env.graphs[0], 4)
        # state为graph的list，用于记录；为运算，需要转为batch并克隆
        state_ = torch_geometric.data.Batch.from_data_list(state).clone()
        state_.x, state_.edge_index = Variable(state_.x.float().to(device)), Variable(state_.edge_index.to(device))
        state_.edge_attr = Variable(state_.edge_attr.to(device))

        # 计算奖励跳步
        cal_freq = int(cal_freq_end + (cal_freq_start - cal_freq_end) * math.exp(-1. * time_step / cal_freq_decay))

        env.cal_freq = cal_freq

        # 游戏步骤循环
        # with tqdm(total=max_ep_len, desc=f'Play: Episode {i_episode}/{max_train_timestep//max_ep_len}', unit='it') as pbar:
        for t in range(1, max_ep_len+1):

            # heat map

            # _, probs = agent.forward(state_, max_size)
            # prob_show = probs.detach().cpu().view(1, -1).numpy()
            #
            # sns.heatmap(data=prob_show, cmap="RdBu_r")
            #
            # plt.show()

            # 智能体产生动作
            actions, action_log_probs = agent.work(state_, env.len_list, max_size, env.forbidden_actions_list, type_=action_type)

            # 环境执行动作
            next_state, reward_list, is_termial, done_list, ids = env.step(actions, t)

            # 数据放入经验池
            actions = actions.split(1, dim=0)
            action_log_probs = action_log_probs.split(1, dim=0)
            if t == max_ep_len:
                pass
            for graph, action, prob, reward, next_graph, id, done in zip(state_.clone().to_data_list(), actions, action_log_probs,
                                                                         reward_list.copy(), next_state, ids,
                                                                         done_list // 10):
                trans = Transition(graph.to(device), action.item(), prob.item(), reward, next_graph, done)
                agent.storeTransition(trans, id)

                ## test: show the change in a round ##
                # energy = get_energy_from_base(graph.y['seq_base'], graph.y['dotB'])
                # hamming = get_distance_from_base(graph.y['seq_base'], graph.y['dotB'])
                # topo = get_topology_distance(graph, env.aim_edge_h_list[id])
                #
                # writer.add_scalar('energy' + str(id), energy, time_step)
                # writer.add_scalar('hamming' + str(id), hamming, time_step)
                # writer.add_scalar('topo' + str(id), topo, time_step)
                #####################################

            done_index = list(np.nonzero(done_list == 10)[0])
            if len(done_index) > 0:
                graphs, remove_ids, dotBs, sequences, energys, distances = env.remove_graph(done_index)

                for graph, remove_id, dotB, sequence, energy, distance in zip(graphs, remove_ids, dotBs, sequences,
                                                                              energys, distances):
                    done_log_f.write(
                        'Episode_{} Graph_{} is removed! Struct: {} | Sequence: {} | Energy: {} | Distance: {}'.format(
                            i_episode, remove_id,
                            dotB,
                            sequence,
                            energy,
                            distance))
                    done_log_f.write('\n')
                    done_log_f.flush()

            # 如果有序列设计完成，则直接训练
            if is_termial:
                env.last_energy_list = np.ones(len(env.graphs_), dtype=float)
                # # 训练
                # loss_a, loss_c = agent.trainStep(i_episode)
                # 更新步数
                time_step += max_ep_len - t + 1
                break

            time_step += 1
            # 记录当前的平均reward
            # current_ep_reward_np += np.array(reward_list)
            current_ep_reward += np.array(reward_list).mean()
            reward_per_step = np.array(reward_list).mean()
            # 更新state
            state_ = torch_geometric.data.Batch.from_data_list(env.graphs).clone()
            state_.x, state_.edge_index = Variable(state_.x.float().to(device)), Variable(state_.edge_index.to(device))
            state_.edge_attr = Variable(state_.edge_attr.to(device))
            # tqdm更新显示
                # pbar.set_postfix({'ratio': env.last_ratio_list.mean()})
                # pbar.update(1)

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        # 从经验池里获取要记录的数据
        final_graphs = [chain[-1].next_state for chain in agent.buffer]
        final_seqs_onehot = [graph.x[:, :4] for graph in final_graphs]
        # final_seqs_base = list(map(seq_onehot2Base, final_seqs_onehot))
        final_seqs_base = [graph.y['seq_base'] for graph in final_graphs]
        final_dotBs = [graph.y['dotB'] for graph in final_graphs]
        final_energy = pool_main.map(get_energy_from_base, final_seqs_base, final_dotBs)
        final_energy = list(final_energy)
        final_distance = pool_main.map(get_distance_from_base, final_seqs_base, final_dotBs)
        final_distance_topo = pool_main.map(get_topology_distance, final_graphs, env.aim_edge_h_list)
        final_distance = list(final_distance)
        final_rewards = []
        for i in range(len(agent.buffer)):
            rewards = [g.reward for g in agent.buffer[i]]
            final_rewards.append(np.array(rewards).sum())

        # 写日志
        if time_step % log_freq == 0:
            for i in range(len(log_f_list)):
                log_f_list[i].write(
                    '{},{},{},{},{},{},{}\n'.format(i_episode, time_step, final_rewards[i],
                                                    final_energy[i],
                                                    final_distance[i], final_dotBs[i],
                                                    final_seqs_base[i]))
                log_f_list[i].flush()

            for i in range(len(act_log_f_list)):
                action_list = [str(t.action) for t in agent.buffer[i]]
                act_str = ','.join(action_list)
                act_log_f_list[i].write(
                    act_str + '\n'
                )
                act_log_f_list[i].flush()

        # 更新
        if time_step % update_timestep == 0:
            loss_a, loss_c = agent.trainStep(i_episode, max_size)
            loss_a = abs(loss_a)
            scheduler_b.step()
            scheduler_c.step()
            scheduler_a.step()

            # 记录到tensorboard
            # loss
            writer.add_scalar('loss_a', loss_a, i_episode)
            writer.add_scalar('loss_c', loss_c, i_episode)

            # reward
            writer.add_scalar('avg_reward', current_ep_reward, i_episode)

            # 网络参数
            for tag_, value in agent.backbone.named_parameters():
                tag_ = "b."+tag_.replace('.', '/')
                writer.add_histogram(tag_, value, i_episode)
                writer.add_histogram(tag_+'/grad', value.grad.data.cpu().numpy(), i_episode)
            for tag_, value in agent.actor.named_parameters():
                tag_ = "a." + tag_.replace('.', '/')
                writer.add_histogram(tag_, value, i_episode)
                if i_episode > actor_freeze_ep:
                    writer.add_histogram(tag_ + '/grad', value.grad.data.cpu().numpy(), i_episode)
            for tag_, value in agent.critic.named_parameters():
                tag_ = "c." + tag_.replace('.', '/')
                writer.add_histogram(tag_, value, i_episode)
                writer.add_histogram(tag_ + '/grad', value.grad.data.cpu().numpy(), i_episode)

            for i in range(len(env.graphs)):
                writer.add_scalar('reward_' + str(i), final_rewards[i], i_episode)
                writer.add_scalar('energy_' + str(i), final_energy[i], i_episode)
                writer.add_scalar('distance_' + str(i), final_distance[i], i_episode)
                writer.add_scalar('distance_tp_' + str(i), final_distance_topo[i], i_episode)
                writer.add_text('sequence_' + str(i), final_seqs_base[i], i_episode)

            agent.clean_buffer()

            # 学习率
            # writer.add_histogram('lr_b', agent.optimizer_b.state_dict()['param_groups'][0]['lr'],
            #                      i_episode)
            writer.add_histogram('lr_a', agent.optimizer_a.state_dict()['param_groups'][0]['lr'],
                                 i_episode)
            writer.add_histogram('lr_c', agent.optimizer_c.state_dict()['param_groups'][0]['lr'],
                                 i_episode)

        # 保存模型
        if time_step % save_model_freq == 0:
            print("--------------------------------------------------------------------------------------------")
            print("saving model at : " + model_dir)
            agent.save(model_dir, i_episode)
            print("model saved")
            print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
            print("--------------------------------------------------------------------------------------------")

    for log_f in log_f_list:
        log_f.close()
    for log_f in act_log_f_list:
        log_f.close()
    # for log_f in place_log_f_list:
    #     log_f.close()
    done_log_f.close()
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == "__main__":
    # root = os.path.dirname(os.path.realpath(__file__))
    # txt_dir = root + '/data/all_data.txt'
    # generate_dataset_for_rl(txt_dir, 10, root + '/data/rl/dataset_10.pt')
    main()