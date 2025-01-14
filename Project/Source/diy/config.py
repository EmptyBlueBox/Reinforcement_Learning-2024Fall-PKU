#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2024 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


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
    REWARD_WEIGHT_DICT = {
        "hp_point": 0.0,
        "tower_hp_point": 0.0,
        "money": 0.0,
        "exp": 0.0,
        "ep_rate": 0.0,
        "death": 0.0,
        "kill": 0.0,
        "last_hit": 0.0,
    }
    # Time decay factor, used in reward_manager
    # 时间衰减因子，在reward_manager中使用
    TIME_SCALE_ARG = 0
    # Evaluation frequency and model save interval configuration, used in workflow
    # 评估频率和模型保存间隔配置，在workflow中使用
    EVAL_FREQ = 10
    MODEL_SAVE_INTERVAL = 1800


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
    DIM_OF_GLOBAL_INFO = [25]


# Configuration related to model and algorithms used
# 模型和算法使用的相关配置
class Config:
    NETWORK_NAME = "network"
    LSTM_TIME_STEPS = 16
    LSTM_UNIT_SIZE = 512
    DATA_SPLIT_SHAPE = [
        810,
        1,
        1,
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
        512,
        512,
    ]
    SERI_VEC_SPLIT_SHAPE = [(725,), (85,)]
    INIT_LEARNING_RATE_START = 0.0001
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

    MIN_POLICY = 0.00001

    TARGET_EMBED_DIM = 32

    data_shapes = [
        [(725 + 85) * 16],
        [16],
        [16],
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
        [512],
        [512],
    ]

    LEGAL_ACTION_SIZE_LIST = LABEL_SIZE_LIST.copy()
    LEGAL_ACTION_SIZE_LIST[-1] = LEGAL_ACTION_SIZE_LIST[-1] * LEGAL_ACTION_SIZE_LIST[0]

    GAMMA = 0.9
    LAMDA = 0.9

    USE_GRAD_CLIP = True
    GRAD_CLIP_RANGE = 0.5

    # The input dimension of samples on the learner from Reverb varies depending on the algorithm used.
    # For instance, the dimension for ppo is 15584,
    # learner上reverb样本的输入维度, 注意不同的算法维度不一样, 比如示例代码中ppo的维度是15584
    # **注意**，此项必须正确配置，应该与definition.py中的NumpyData2SampleData函数数据对齐，否则可能报样本维度错误
    SAMPLE_DIM = 15584
