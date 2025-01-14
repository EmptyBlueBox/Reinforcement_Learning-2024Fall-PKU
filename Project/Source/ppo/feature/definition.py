#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2024 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from kaiwu_agent.utils.common_func import create_cls, Frame, attached
import numpy as np
import collections
from ppo.config import Config
import random
import itertools
import os
import json
from collections import OrderedDict

from ppo.config import GameConfig

SampleData = create_cls("SampleData", npdata=None)

ObsData = create_cls("ObsData", feature=None, legal_action=None, lstm_cell=None, lstm_hidden=None)

ActData = create_cls(
    "ActData",
    action=None,
    d_action=None,
    prob=None,
    value=None,
    lstm_cell=None,
    lstm_hidden=None,
)

NONE_ACTION = [0, 15, 15, 15, 15, 0]    # ？


# Loop through camps, shuffling camps before each major loop
# 循环返回camps, 每次大循环前对camps进行shuffle
def _lineup_iterator_shuffle_cycle(camps):  # 生成器函数，它对输入的阵营进行打乱，然后无限循环地返回随机顺序的阵营。
    while True:
        random.shuffle(camps)   #shuffle打乱顺序
        for camp in camps:
            yield camp

# 508对一切
# 133对508
# 199对133，508

# Specify single-side multi-agent lineups, looping through all pairwise combinations
# 指定单边多智能体阵容，两两组合循环
def lineup_iterator_roundrobin_camp_heroes(camp_heroes=None):
    if not camp_heroes:
        raise Exception(f"camp_heroes is empty")

    try:    #异常处理
        valid_ids = [133, 199, 508]
        for camp in camp_heroes:
            hero_id = camp[0]["hero_id"]
            if hero_id not in valid_ids:
                raise Exception(f"hero_id {hero_id} not valid, valid is {valid_ids}")
    except Exception as e:
        raise Exception(f"check hero valid, exception is {str(e)}")

    camps = []  #形成组队表格
    for lineups in itertools.product(camp_heroes, camp_heroes):
        camp = []
        for lineup in lineups:
            camp.append(lineup)
        # if 508 in camp:
        #     camps.append(camp)
        # if 133 in camp and 199 in camp:
        #     camps.append(camp)
        camps.append(camp)
    return _lineup_iterator_shuffle_cycle(camps)


@attached
def sample_process(collector):
    return collector.sample_process()

bullet_type_set = {     # ？ 这三兄贵是怎么作用的
    133: set(),
    199: set(),
    508: set(),
}
bullet_id_set = {
    133: set(),
    199: set(),
    508: set(),
}
buff_set = set()

# Create the sample for the current frame
# 创建当前帧的样本
def build_frame(agent, state_dict):
    obs_data, act_data = agent.obs_data, agent.act_data

    # get is_train
    is_train = False                                        # 用于区分智能体是否在训练模式下，通过判断英雄的生命值 (hero_hp) 来确定是否继续训练。
    frame_state = state_dict["frame_state"]                 # 获取当前帧的状态信息
    hero_list = frame_state["hero_states"]                  # 获取当前帧的英雄状态信息
    frame_no = frame_state["frameNo"]                       # 获取当前帧的帧号
    npc_list = frame_state["npc_states"]                    # 获取当前帧的NPC状态信息
    main_hero_player_id = state_dict['player_id']           # 获取主英雄的player_id
    main_hero = None                                        
    enemy_hero = None
    for hero in hero_list:                                  # 提取英雄阵营、生命值和玩家 ID
        hero_camp = hero["actor_state"]["camp"]
        hero_hp = hero["actor_state"]["hp"]

        hero_player_id = hero["player_id"]
        if hero_player_id == main_hero_player_id:           # 判断是否是主英雄。如果是，将其信息存储到 main_hero；如果不是，则存储到 enemy_hero。
            main_hero = hero
        else:
            enemy_hero = hero

        if hero_camp == agent.hero_camp:                    # 如果英雄阵营与智能体的阵营相同，则将其生命值存储到 hero_hp。
            is_train = True if hero_hp > 0 else False

    if obs_data.feature is not None:
        feature_vec = np.array(obs_data.feature)
    else:
        feature_vec = np.array(state_dict["observation"])
    
    # print(main_hero['actor_state']['values']['mov_spd'])

    # reward_sum
    reward = state_dict["reward"]["reward_sum"]

    # farming, KDA, damage, pushing, other(win)
    # multi_reward = [
    #     state_dict["reward"]["money"]*RewardWeight["money"] + state_dict["reward"]["exp"]*RewardWeight["exp"] + state_dict["reward"]["last_hit"]*RewardWeight["last_hit"],
    #     state_dict["reward"]["death"]*RewardWeight["death"] + state_dict["reward"]["kill"]*RewardWeight["kill"],
    #     state_dict["reward"]["hp_point"]*RewardWeight["hp_point"] + state_dict["reward"]["ep_rate"]*RewardWeight["ep_rate"],
    #     state_dict["reward"]["tower_hp_point"]*RewardWeight["tower_hp_point"],
    #     state_dict["reward"]["forward"]*RewardWeight["forward"] + state_dict["reward"]["close_to_cake"]*RewardWeight["close_to_cake"],
    # ]
        
    # assert round(sum(multi_reward), 4) == round(state_dict["reward"]["reward_sum"], 4)

    sub_action_mask = state_dict["sub_action_mask"]     #获取子动作掩码

    prob, value, action = act_data.prob, act_data.value, act_data.action
    lstm_cell, lstm_hidden = act_data.lstm_cell, act_data.lstm_hidden       #在时间序列的强化学习模型中，LSTM（长短期记忆网络）可以帮助模型记住之前的状态。
                                                                            #这里从动作数据中提取了 LSTM 的 cell 和 hidden 状态，用于后续的序列决策。

    legal_action = _update_legal_action(state_dict["legal_action"], action)     #更新合法的动作空间
    #multi-head value
    # frame = Frame(
    #     frame_no=frame_no,
    #     feature=feature_vec.reshape([-1]),
    #     legal_action=legal_action.reshape([-1]),
    #     action=action,
    #     reward=reward,
    #     multi_reward=multi_reward,
    #     multi_reward_sum = np.zeros((5,)),
    #     reward_sum=0,
    #     value=value[0],
    #     next_value=np.zeros((5,)),
    #     advantage=np.zeros((5,)),
    #     prob=prob,
    #     sub_action=sub_action_mask[action[0]],
    #     lstm_info=np.concatenate([lstm_cell.flatten(), lstm_hidden.flatten()]).reshape([-1]),
    #     is_train=False if action[0] < 0 else is_train,
    # )
    frame = Frame(
        frame_no=frame_no,                                  
        feature=feature_vec.reshape([-1]),
        legal_action=legal_action.reshape([-1]),
        action=action,
        reward=reward,
        reward_sum=0,
        value=value.flatten()[0],
        next_value=0,
        advantage=0,
        prob=prob,
        sub_action=sub_action_mask[action[0]],
        lstm_info=np.concatenate([lstm_cell.flatten(), lstm_hidden.flatten()]).reshape([-1]),
        is_train=False if action[0] < 0 else is_train,
    )
    return frame

# Construct legal_action based on the actual action taken
# 根据实际采用的action，构建legal_action
def _update_legal_action(original_la, action):  #这个函数用于根据智能体执行的动作更新合法的动作空间。不同的状态允许执行不同的动作
    target_size = Config.LABEL_SIZE_LIST[-1]
    top_size = Config.LABEL_SIZE_LIST[0]
    original_la = np.array(original_la)
    fix_part = original_la[: -target_size * top_size]
    target_la = original_la[-target_size * top_size :]
    target_la = target_la.reshape([top_size, target_size])[action[0]]
    return np.concatenate([fix_part, target_la], axis=0)


class FrameCollector:
    def __init__(self, num_agents):
        self._data_shapes = Config.data_shapes
        self._LSTM_FRAME = Config.LSTM_TIME_STEPS

        self.num_agents = num_agents
        self.rl_data_map = [collections.OrderedDict() for _ in range(num_agents)]
        self.m_replay_buffer = [[] for _ in range(num_agents)]

        # load config from config file
        self.gamma = Config.GAMMA
        self.lamda = Config.LAMDA
        self.last_data_size = 8
        self.last_sample_batch = [[] for _ in range(2)]
        self.last_sample_lstm = [[] for _ in range(2)]
        self.is_first_frame = [1, 1]

    def reset(self, num_agents):
        self.num_agents = num_agents
        self.rl_data_map = [collections.OrderedDict() for _ in range(self.num_agents)]
        self.m_replay_buffer = [[] for _ in range(self.num_agents)]

    def save_frame(self, rl_data_info, agent_id):
        # samples must saved by frame_no order
        # 样本必须按帧号顺序保存
        reward = self._clip_reward(rl_data_info.reward)

        # update last frame's next_value
        if len(self.rl_data_map[agent_id]) > 0:
            last_key = list(self.rl_data_map[agent_id].keys())[-1]
            last_rl_data_info = self.rl_data_map[agent_id][last_key]
            last_rl_data_info.next_value = rl_data_info.value
            last_rl_data_info.reward = reward

        rl_data_info.reward = 0
        self.rl_data_map[agent_id][rl_data_info.frame_no] = rl_data_info

    def save_last_frame(self, reward, agent_id):
        if len(self.rl_data_map[agent_id]) > 0:
            last_key = list(self.rl_data_map[agent_id].keys())[-1]
            last_rl_data_info = self.rl_data_map[agent_id][last_key]
            # last_rl_data_info.next_value = np.zeros(5)
            last_rl_data_info.next_value = 0
            last_rl_data_info.reward = reward


    def sample_process(self):
        self._calc_reward()
        self._format_data()
        return self.m_replay_buffer

    def _calc_reward(self):
        """
        Calculate cumulated reward and advantage with GAE.
        reward_sum: used for value loss
        advantage: used for policy loss
        V(s) here is a approximation of target network
        """
        """
        计算累积奖励和优势函数（GAE）。
        reward_sum: 用于值损失
        advantage: 用于策略损失
        V(s) 这里是目标网络的近似值
        """
        # multi-head value
        # for i in range(self.num_agents):
        #     reversed_keys = list(self.rl_data_map[i].keys())
        #     reversed_keys.reverse()
        #     last_gae = 0.0
        #     gae = np.zeros((5,))
        #     for j in reversed_keys:
        #         rl_info = self.rl_data_map[i][j]
        #         delta = -np.array(rl_info.value) + np.array(rl_info.multi_reward) + self.gamma * np.array(rl_info.next_value)
        #         gae = gae * self.gamma * self.lamda + delta
        #         rl_info.advantage = gae
        #         rl_info.multi_reward_sum = gae + np.array(rl_info.value)
        for i in range(self.num_agents):
            reversed_keys = list(self.rl_data_map[i].keys())
            reversed_keys.reverse()
            gae, last_gae = 0.0, 0.0    # ? 没使用到上一次的gae
            for j in reversed_keys:
                rl_info = self.rl_data_map[i][j]
                delta = -rl_info.value + rl_info.reward + self.gamma * rl_info.next_value
                gae = gae * self.gamma * self.lamda + delta
                rl_info.advantage = gae
                rl_info.reward_sum = gae + rl_info.value        

    # For every LSTM_TIME_STEPS samples, concatenate 1 LSTM state
    # 每LSTM_TIME_STEPS个样本，需要拼接1个lstm状态
    def _reshape_lstm_batch_sample(self, sample_batch, sample_lstm):
        # LSTM网络在处理序列数据时，需要将多个时间步长的数据一起处理。这个函数负责将时间步长样本和LSTM状态拼接在一起，形成一个完整的样本用于训练。

        sample = np.zeros([np.prod(sample_batch.shape) + np.prod(sample_lstm.shape)])
        idx, s_idx = 0, 0

        sample[-sample_lstm.shape[0] :] = sample_lstm
        for split_shape in self._data_shapes[:-2]:
            one_shape = split_shape[0] // self._LSTM_FRAME
            sample[s_idx : s_idx + split_shape[0]] = sample_batch[:, idx : idx + one_shape].reshape([-1])
            idx += one_shape
            s_idx += split_shape[0]
        return SampleData(npdata=sample.astype(np.float32))

    # Create the sample for the current frame
    # 根据LSTM_TIME_STEPS，组合送入样本池的样本
    def _format_data(self):
          
        sample_one_size = np.sum(self._data_shapes[:-2]) // self._LSTM_FRAME    # 计算了每个样本的特征总维度
        sample_lstm_size = np.sum(self._data_shapes[-2:])                       # 计算了用于 LSTM 的特定维度大小
        sample_batch = np.zeros([self._LSTM_FRAME, sample_one_size])            # 存储每个样本的特征
        first_frame_no = -1                                                     # 保存第一帧的编号

        for i in range(self.num_agents):                         # 用来存储 LSTM 格式化的数据
            have_bullet = False
            sample_lstm = np.zeros([sample_lstm_size])

            cnt = 0
            for j in self.rl_data_map[i]:
                rl_info = self.rl_data_map[i][j]
                if cnt == 0:
                    # lstm cell & hidden
                    first_frame_no = rl_info.frame_no

                # serilize one frames
                idx, dlen = 0, 0

                # vec_data  提取特征向量 rl_info.feature 并填充到 sample_batch 中。
                dlen = rl_info.feature.shape[0]
                sample_batch[cnt, idx : idx + dlen] = rl_info.feature
                idx += dlen

                # 判断是否有bullet
                have_bullet = ((sum(rl_info.feature[726:756]) != 0) or have_bullet)
                print(have_bullet)

                # legal_action
                dlen = rl_info.legal_action.shape[0]
                sample_batch[cnt, idx : idx + dlen] = rl_info.legal_action
                idx += dlen

                # reward_sum & advantage    将 reward_sum 和 advantage 分别填充到样本。
                # sample_batch[cnt, idx: idx + 5] = rl_info.multi_reward_sum
                # idx += 5
                # sample_batch[cnt, idx: idx + 5] = rl_info.advantage
                # idx += 5
                sample_batch[cnt, idx] = rl_info.reward_sum
                idx += 1
                sample_batch[cnt, idx] = rl_info.advantage
                idx += 1

                # labels                    提取 action（长度为6）填充。
                dlen = 6
                sample_batch[cnt, idx : idx + dlen] = rl_info.action
                idx += dlen

                # probs (neg log pi->prob)  计算并填充每个动作的概率 prob。
                for p in rl_info.prob:
                    dlen = len(p)
                    # p = np.exp(-nlp)
                    # p = p / np.sum(p)
                    sample_batch[cnt, idx : idx + dlen] = p
                    idx += dlen

                # sub_action                将 sub_action（长度为6）填充。
                dlen = 6
                sample_batch[cnt, idx : idx + dlen] = rl_info.sub_action
                idx += dlen

                # is_train                  将 is_train（长度为1）填充。
                sample_batch[cnt, idx] = rl_info.is_train
                idx += 1

                assert idx == sample_one_size, "Sample check failed, {}/{}".format(idx, sample_one_size)

                cnt += 1
                if cnt == self.last_data_size and self.is_first_frame[i] == 0:
                    self.last_sample_lstm[i] = rl_info.lstm_info
                if cnt == self._LSTM_FRAME:
                    cnt = 0
                    if self.is_first_frame[i] == 0:
                        input = np.vstack((self.last_sample_batch[i], sample_batch[:self.last_data_size]))
                        sample = self._reshape_lstm_batch_sample(input, self.last_sample_lstm[i])
                        self.m_replay_buffer[i].append(sample)
                    sample = self._reshape_lstm_batch_sample(sample_batch, sample_lstm)
                    self.m_replay_buffer[i].append(sample)
                    if have_bullet:
                        self.m_replay_buffer[i].append(sample)
                    self.last_sample_batch[i] = sample_batch[-self.last_data_size:]
                    if self.is_first_frame[i] == 1:
                        self.is_first_frame[i] = 0
                    sample_lstm = rl_info.lstm_info
                    have_bullet = False

    def _clip_reward(self, reward, max=100, min=-100):  #奖励裁剪函数，用于将奖励限制在指定的区间范围 ? 这个范围是怎么确定的
        if reward > max:
            reward = max
        elif reward < min:
            reward = min
        return reward

    def __len__(self):
        return max([len(agent_samples) for agent_samples in self.rl_data_map])


# SampleData <----> NumpyData
@attached
def SampleData2NumpyData(g_data):
    return g_data.npdata


@attached
def NumpyData2SampleData(s_data):
    return SampleData(npdata=s_data)