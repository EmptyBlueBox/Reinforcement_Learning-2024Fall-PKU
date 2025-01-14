#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2024 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""
buff_info_len = 222

class GameConfig:

    """
    Specify the training lineup in CAMP_HEROES. The battle lineup will be paired in all possible combinations.
    To train a single agent, comment out the other agents.
    1. 133 DiRenjie
    2. 199 Arli
    3. 508 Garo
    """

    """
    在CAMP_HEROES中指定训练阵容, 对战阵容会两两组合, 训练单智能体则注释其他智能体。此配置会在阵容生成器中使用
    1. 133 狄仁杰
    2. 199 公孙离
    3. 508 伽罗
    """
    CAMP_HEROES = [
        [{"hero_id": 133}],
        [{"hero_id": 199}],
        [{"hero_id": 508}],
    ]
    # Set the weight of each reward item and use it in reward_manager
    # 设置各个回报项的权重，在reward_manager中使用
    REWARD_WEIGHT_DICT_1 = {
        "hp_point": 3.5,
        "tower_hp_point": 10,
        "money": 0.009,
        "exp": 0.006,
        "ep_rate": 0.1,
        "death": -1.0,
        "kill": 0.3,
        "last_hit": 0.5,
        "forward": 0.01,
        "close_to_cake": 0,
        "extra_mov_spd": 0,
        "kill_monster": 0,
        "hit_target": 3.0,
        "kiting": 0,
        "hit_by_organ": -1.5,
    }
    REWARD_WEIGHT_CHANGE_POINT_1_2 = 3600
    REWARD_WEIGHT_DICT_2 = {
        "hp_point": 3,
        "tower_hp_point": 30,
        "money": 0.006,
        "exp": 0.003,
        "ep_rate": 0.09,
        "death": -1.0,
        "kill": -0.4,
        "last_hit": 0.4,
        "forward": 0.0085,
        "close_to_cake": 0,
        "extra_mov_spd": 0,
        "kill_monster": 0,
        "hit_target": 2.0,
        "kiting": 0,
        "hit_by_organ": -1.0,
    }
    REWARD_WEIGHT_CHANGE_POINT_2_3 = 12000
    REWARD_WEIGHT_DICT_3 = {
        "hp_point": 1,
        "tower_hp_point": 15,
        "money": 0.0015,
        "exp": 0.0005,
        "ep_rate": 0.03,
        "death": -0.4,
        "kill": -0.15,
        "last_hit": 0.05,
        "forward": 0.003,
        "close_to_cake": 0,
        "extra_mov_spd": 0,
        "kill_monster": 0,
        "hit_target": 0.5,
        "kiting": 0,
        "hit_by_organ": -0.35,
    }   
    REWARD_WEIGHT_DICT_4 = {
        "hp_point": 0.5,
        "tower_hp_point": 10,
        "money": 0.0005,
        "exp": 0,
        "ep_rate": 0.01,
        "death": -0.25,
        "kill": -0.1,
        "last_hit": 0,
        "forward": 0.00215,
        "close_to_cake": 0,
        "extra_mov_spd": 0,
        "kill_monster": 0,
        "hit_target": 0.25,
        "kiting": 0,
        "hit_by_organ": -0.25,
    }   


    TARGET_REWARD_WEIGHT_DICT_1 = {
        "hp_point": 4,
        "tower_hp_point": 15,
        "money": 0.009,
        "exp": 0.006,
        "ep_rate": 0.01,
        "death": -1.0,
        "kill": 1.0,
        "last_hit": 0.5,
        "forward": 0.01,
        "close_to_cake": 0.01,
        "extra_mov_spd": 0.003,
        "kill_monster": 0.4,
        "hit_target": 3.0,
        "kiting": 0.05,
        "hit_by_organ": -1.5,
    }
    TARGET_REWARD_WEIGHT_CHANGE_POINT_1_2 = 3600
    TARGET_REWARD_WEIGHT_DICT_2 = {
        "hp_point": 5,
        "tower_hp_point": 40,
        "money": 0.005,
        "exp": 0.003,
        "ep_rate": 0.006,
        "death": -0.8,
        "kill": -0.4,
        "last_hit": 0.4,
        "forward": 0.0085,
        "close_to_cake": 0.0085,
        "extra_mov_spd": 0.0038,
        "kill_monster": 0.5,
        "hit_target": 2.0,
        "kiting": 0.042,
        "hit_by_organ": -1.0,
    }
    TARGET_REWARD_WEIGHT_CHANGE_POINT_2_3 = 12000
    TARGET_REWARD_WEIGHT_DICT_3 = {
        "hp_point": 2.5,
        "tower_hp_point": 20,
        "money": 0.0015,
        "exp": 0.0005,
        "ep_rate": 0.001,
        "death": -0.6,
        "kill": -0.3,
        "last_hit": 0.05,
        "forward": 0.003,
        "close_to_cake": 0.003,
        "extra_mov_spd": 0.0009,
        "kill_monster": 0.15,
        "hit_target": 1.0,
        "kiting": 0.015,
        "hit_by_organ": -0.35,
    }   
    TARGET_REWARD_WEIGHT_DICT_4 = {
        "hp_point": 1.25,
        "tower_hp_point": 12.5,
        "money": 0.0005,
        "exp": 0,
        "ep_rate": 0,
        "death": -0.4,
        "kill": -0.2,
        "last_hit": 0,
        "forward": 0.00215,
        "close_to_cake": 0.00215,
        "extra_mov_spd": 0.0006,
        "kill_monster": 0.08,
        "hit_target": 0.5,
        "kiting": 0.005,
        "hit_by_organ": -0.25,
    }   
    # Time decay factor, used in reward_manager
    # 时间衰减因子，在reward_manager中使用
    TIME_SCALE_ARG = 12000
    # Evaluation frequency and model save interval configuration, used in workflow
    # 评估频率和模型保存间隔配置，在workflow中使用
    EVAL_FREQ = 10
    MODEL_SAVE_INTERVAL = 600


# Dimension configuration, used when building the model
# 维度配置，构建模型时使用
class DimConfig:
    # main camp soldier
    DIM_OF_SOLDIER_1_10 = [18, 18, 18, 18]
    # enemy camp soldier
    DIM_OF_SOLDIER_11_20 = [18, 18, 18, 18]
    # main camp organ
    DIM_OF_ORGAN_1_2 = [18, 18]
    # enemy camp organ
    DIM_OF_ORGAN_3_4 = [18, 18]
    # main camp hero
    DIM_OF_HERO_FRD = [235]
    # enemy camp hero
    DIM_OF_HERO_EMY = [235]
    # public hero info
    DIM_OF_HERO_MAIN = [14]
    # global info
    DIM_OF_GLOBAL_INFO = [5]
    # hero_id info
    DIM_OF_HERO_ID_INFO = [6]
    # monster info
    DIM_OF_MONSTER_INFO = [14]
    # cake info
    DIM_OF_CAKE_INFO = [2]
    # bullet info
    DIM_OF_BULLET_INFO = [30]
    # buff info
    DIM_OF_BUFF_INFO = [buff_info_len]
    # command info
    DIM_OF_COMMAND_INFO = [54]
    # target info
    DIM_OF_TARGET_INFO = [28]


# Configuration related to model and algorithms used
# 模型和算法使用的相关配置
class Config:
    NETWORK_NAME = "network"
    LSTM_TIME_STEPS = 16
    LSTM_UNIT_SIZE = 1024
    LSTM_REAL_SIZE = 256
    DATA_SPLIT_SHAPE = [
        810+114+buff_info_len,
        1, # multi-head value 5
        1, # multi-head value 5
        1,
        1,
        1,
        1,
        1,
        1,
        12,
        16,
        16,
        16,
        16,
        9,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        256,
        256,
    ]
    SERI_VEC_SPLIT_SHAPE = [(725+114+buff_info_len,), (85,)]
    INIT_LEARNING_RATE_START = 5e-5
    BETA_START = 0.025
    LOG_EPSILON = 1e-6
    LABEL_SIZE_LIST = [12, 16, 16, 16, 16, 9]
    IS_REINFORCE_TASK_LIST = [
        True,
        True,
        True,
        True,
        True,
        True,
    ]  # means each task whether need reinforce

    CLIP_PARAM = 0.2

    MIN_POLICY = 1e-5

    TARGET_EMBED_DIM = 86

    data_shapes = [
        [(725 + 114 + buff_info_len + 85) * 16],
        [16], # multi-head value 80
        [16], # multi-head value 80
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [192],
        [256],
        [256],
        [256],
        [256],
        [144],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [256],
        [256],
    ]

    LEGAL_ACTION_SIZE_LIST = LABEL_SIZE_LIST.copy()
    LEGAL_ACTION_SIZE_LIST[-1] = LEGAL_ACTION_SIZE_LIST[-1] * LEGAL_ACTION_SIZE_LIST[0]

    GAMMA = 0.996
    LAMDA = 0.95

    USE_GRAD_CLIP = True
    GRAD_CLIP_RANGE = 0.5

    # The input dimension of samples on the learner from Reverb varies depending on the algorithm used.
    # For instance, the dimension for ppo is 15584,
    # learner上reverb样本的输入维度, 注意不同的算法维度不一样, 比如示例代码中ppo的维度是15584
    # **注意**，此项必须正确配置，应该与definition.py中的NumpyData2SampleData函数数据对齐，否则可能报样本维度错误
    SAMPLE_DIM = 15584+114*16+buff_info_len*16-256*2
    # SAMPLE_DIM = np.sum(data_shapes)