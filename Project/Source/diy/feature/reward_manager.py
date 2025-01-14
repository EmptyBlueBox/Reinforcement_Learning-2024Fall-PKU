#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2024 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import math
from diy.config import GameConfig

# Used to record various reward information
# 用于记录各个奖励信息
class RewardStruct:
    def __init__(self, m_weight=0.0):
        self.cur_frame_value = 0.0
        self.last_frame_value = 0.0
        self.value = 0.0
        self.weight = m_weight
        self.min_value = -1
        self.is_first_arrive_center = True


# Used to initialize various reward information
# 用于初始化各个奖励信息
def init_calc_frame_map():
    calc_frame_map = {}
    for key, weight in GameConfig.REWARD_WEIGHT_DICT.items():
        calc_frame_map[key] = RewardStruct(weight)
    return calc_frame_map


class GameRewardManager:
    def __init__(self, main_hero_runtime_id):
        self.main_hero_player_id = main_hero_runtime_id
        self.main_hero_camp = -1
        self.main_hero_hp = -1
        self.main_hero_organ_hp = -1
        self.m_reward_value = {}
        self.m_last_frame_no = -1
        self.m_cur_calc_frame_map = init_calc_frame_map()
        self.m_main_calc_frame_map = init_calc_frame_map()
        self.m_enemy_calc_frame_map = init_calc_frame_map()
        self.m_init_calc_frame_map = {}
        self.time_scale_arg = GameConfig.TIME_SCALE_ARG
        self.m_main_hero_config_id = -1
        self.m_each_level_max_exp = {}

    # Used to initialize the maximum experience value for each agent level
    # 用于初始化智能体各个等级的最大经验值
    def init_max_exp_of_each_hero(self):
        self.m_each_level_max_exp.clear()
        self.m_each_level_max_exp[1] = 160
        self.m_each_level_max_exp[2] = 298
        self.m_each_level_max_exp[3] = 446
        self.m_each_level_max_exp[4] = 524
        self.m_each_level_max_exp[5] = 613
        self.m_each_level_max_exp[6] = 713
        self.m_each_level_max_exp[7] = 825
        self.m_each_level_max_exp[8] = 950
        self.m_each_level_max_exp[9] = 1088
        self.m_each_level_max_exp[10] = 1240
        self.m_each_level_max_exp[11] = 1406
        self.m_each_level_max_exp[12] = 1585
        self.m_each_level_max_exp[13] = 1778
        self.m_each_level_max_exp[14] = 1984

    def result(self, frame_data):
        return self.m_reward_value

    # Calculate the value of each reward item in each frame
    # 计算每帧的每个奖励子项的信息
    def set_cur_calc_frame_vec(self, cul_calc_frame_map, frame_data, camp):
        return

    # Calculate the total amount of experience gained using agent level and current experience value
    # 用智能体等级和当前经验值，计算获得经验值的总量
    def calculate_exp_sum(self, this_hero_info):
        return

    # Use the values obtained in each frame to calculate the corresponding reward value
    # 用每一帧得到的奖励子项信息来计算对应的奖励值
    def get_reward(self, frame_data, reward_dict):
        reward_dict.clear()
        reward_sum, weight_sum = 0.0, 0.0
        for reward_name, reward_struct in self.m_cur_calc_frame_map.items():
            pass
        reward_dict["reward_sum"] = reward_sum
