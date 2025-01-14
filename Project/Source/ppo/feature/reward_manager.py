#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2024 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import time
import math
from ppo.config import GameConfig
from dataclasses import dataclass
from typing import TypedDict, Dict, List

# 将频繁使用的字典值缓存为常量
HERO_CONFIG_IDS = {133, 199, 508}  # 支持的英雄ID集合

# 优化 hero_money_limit 字典访问
def get_hero_money_limit(hero_id):
    limits = {
        133: 6200,  # 狄仁杰
        199: 8900,  # 公孙离
        508: 12400, # 伽罗
    }
    return limits.get(hero_id, 0)

# 优化技能奖励计算
def get_skill_reward_weight(hero_id, skill_id):
    weights = {
        133: {0: 0.01, 1: 0.3, 2: 0.1, 3: 1.0},
        199: {0: 0.01, 1: 0.0, 2: 0.3, 3: 0.5},
        508: {0: 0.01, 1: 0.0, 2: 0.2, 3: 0.1}
    }
    return weights.get(hero_id, {}).get(skill_id, 0.0)

usingProgressiveRewardAdjustment = True #选择是否使用阶段性奖励调整
rewardChangeInterval = 7200   #奖励变化间隔
rewardChangeStep = 30   #步长？学习率？
rewardChangeTimeList = [timeNode for timeNode in range(0, rewardChangeInterval, rewardChangeStep)]    #奖励变化时间列表 #为什么这样子设置？

def get_progressive_reward_map(old_map, new_map, timeNode):   #奖励映射调整函数 - 根据时间节点逐步调整奖励映射，使奖励更加平滑地变化。
    for key in GameConfig.REWARD_WEIGHT_DICT_1.keys():
        new_map[key].weight = old_map[key].weight + (new_map[key].weight - old_map[key].weight) * timeNode / rewardChangeInterval # 越到后期学的越多？
    return new_map


# Used to record various reward information
# 用于记录各个奖励信息
@dataclass
class RewardStruct:
    weight: float = 0.0
    cur_frame_value: float = 0.0
    last_frame_value: float = 0.0
    value: float = 0.0
    min_value: float = -1
    is_first_arrive_center: bool = True


# Used to initialize various reward information
# 用于初始化各个奖励信息
def init_calc_frame_map():
    calc_frame_map = {}
    for key, weight in GameConfig.REWARD_WEIGHT_DICT_1.items(): #这里的weight是什么？
        calc_frame_map[key] = RewardStruct(weight)
    return calc_frame_map

def change_calc_frame_map(calc_frame_map, stage, frame_no):  #根据游戏的当前阶段（stage）和帧号（frame_no）调整奖励权重映射
    #它将根据已过的帧数（frame_no）在 GameConfig.REWARD_WEIGHT_DICT_n 和 GameConfig.REWARD_WEIGHT_DICT_n+1 之间插值来更新权重。
    if stage == 1:
        for key, weight in GameConfig.REWARD_WEIGHT_DICT_1.items():
            calc_frame_map[key].weight = weight + (GameConfig.REWARD_WEIGHT_DICT_2[key] - weight)*(frame_no-0)/(3600-0)
        return calc_frame_map
    elif stage == 2:
        for key, weight in GameConfig.REWARD_WEIGHT_DICT_2.items():
            calc_frame_map[key].weight = weight + (GameConfig.REWARD_WEIGHT_DICT_3[key] - weight)*(frame_no-3600)/(12000-3600)
        return calc_frame_map
    elif stage == 3:
        for key, weight in GameConfig.REWARD_WEIGHT_DICT_3.items():
            calc_frame_map[key].weight = weight + (GameConfig.REWARD_WEIGHT_DICT_4[key] - weight)*(frame_no-12000)/(20000-12000)
        return calc_frame_map

def change_target_calc_frame_map(calc_frame_map, stage, frame_no):  #与 change_calc_frame_map 类似，但它用于目标奖励权重的调整
    if stage == 1:
        for key, weight in GameConfig.TARGET_REWARD_WEIGHT_DICT_1.items():
            calc_frame_map[key].weight = weight + (GameConfig.REWARD_WEIGHT_DICT_2[key] - weight)*(frame_no-0)/(3600-0)
        return calc_frame_map
    elif stage == 2:
        for key, weight in GameConfig.TARGET_REWARD_WEIGHT_DICT_2.items():
            calc_frame_map[key].weight = weight + (GameConfig.REWARD_WEIGHT_DICT_3[key] - weight)*(frame_no-3600)/(12000-3600)
        return calc_frame_map
    elif stage == 3:
        for key, weight in GameConfig.TARGET_REWARD_WEIGHT_DICT_3.items():
            calc_frame_map[key].weight = weight + (GameConfig.REWARD_WEIGHT_DICT_4[key] - weight)*(frame_no-12000)/(20000-12000)
        return calc_frame_map


class GameRewardManager:
    def __init__(self, main_hero_runtime_id):
        self.time_step = -1                                             # 时间步

        self.stage_1_2 = GameConfig.REWARD_WEIGHT_CHANGE_POINT_1_2      # 表示第一阶段和第二阶段之间奖励权重变化的点
        self.stage_2_3 = GameConfig.REWARD_WEIGHT_CHANGE_POINT_2_3      

        self.stage_1_weight = GameConfig.REWARD_WEIGHT_DICT_1           # 第一阶段的奖励权重字典
        self.stage_2_weight = GameConfig.REWARD_WEIGHT_DICT_2           
        self.stage_3_weight = GameConfig.REWARD_WEIGHT_DICT_3           

        self.main_hero_player_id = main_hero_runtime_id                 # 主英雄的玩家ID
        self.main_hero_camp = -1                                        # 主英雄的阵营
        self.main_hero_hp = -1                                          # 主英雄的生命值           
        self.main_hero_organ_hp = -1                                    # 主英雄的法力值
        self.m_reward_value = {}                                        # 用于存储每一帧的奖励值的字典。
        self.m_last_frame_no = -1                                       # 上一帧的帧号

        self.m_cur_calc_frame_map = init_calc_frame_map()               # 当前帧的奖励映射
        self.m_main_calc_frame_map = init_calc_frame_map()              # 主英雄的奖励映射
        self.m_enemy_calc_frame_map = init_calc_frame_map()             # 敌方英雄的奖励映射
        self.m_init_calc_frame_map = {}                                 # 初始化的奖励映射

        self.time_scale_arg = GameConfig.TIME_SCALE_ARG                 # 时间缩放参数
        
        self.m_main_hero_config_id = -1                                 # 主英雄的配置ID
        self.m_each_level_max_exp = {}                                  # 每个等级的最大经验值
        self.start_time = 1727897050                                    # 这个时间是用来干什么的？

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
        self.init_max_exp_of_each_hero()
        self.frame_data_process(frame_data)
        self.get_reward(frame_data, self.m_reward_value)

        frame_no = frame_data["frameNo"]
        # if self.time_scale_arg > 0:
        #     for key in self.m_reward_value:
        #         if key == 'towel_hp_point':
        #             self.m_reward_value[key] *= math.pow(0.8, 1.0 * frame_no / self.time_scale_arg)
        #         else:
        #             self.m_reward_value[key] *= math.pow(0.6, 1.0 * frame_no / self.time_scale_arg)
        hero_list = frame_data["hero_states"]
        for hero in hero_list:
            hero_camp = hero["actor_state"]["camp"]
            if hero_camp == self.main_hero_camp:
                main_hero = hero
            else:
                enemy_hero = hero
        if main_hero['level'] < 4:
            self.m_reward_value['exp'] *= 3

        return self.m_reward_value

    # Calculate the value of each reward item in each frame
    # 计算每帧的每个奖励子项的信息
    def set_cur_calc_frame_vec(self, cul_calc_frame_map, frame_data, camp):

        # Get both agents
        # 获取双方智能体
        main_hero, enemy_hero = None, None
        hero_list = frame_data["hero_states"]
        for hero in hero_list:
            hero_camp = hero["actor_state"]["camp"]
            if hero_camp == camp:
                main_hero = hero
            else:
                enemy_hero = hero
        main_hero_hp = main_hero["actor_state"]["hp"]
        main_hero_max_hp = main_hero["actor_state"]["max_hp"]
        main_hero_ep = main_hero["actor_state"]["values"]["ep"]
        main_hero_max_ep = main_hero["actor_state"]["values"]["max_ep"]

        # Get both defense towers
        # 获取双方防御塔
        main_tower, main_spring, enemy_tower, enemy_spring = None, None, None, None
        npc_list = frame_data["npc_states"]
        for organ in npc_list:
            organ_camp = organ["camp"]
            organ_subtype = organ["sub_type"]
            if organ_camp == camp:
                if organ_subtype == "ACTOR_SUB_TOWER":  # 21 is ACTOR_SUB_TOWER, normal tower
                    main_tower = organ
                elif organ_subtype == "ACTOR_SUB_CRYSTAL":  # 24 is ACTOR_SUB_CRYSTAL, base crystal
                    main_spring = organ
            else:
                if organ_subtype == "ACTOR_SUB_TOWER":  # 21 is ACTOR_SUB_TOWER, normal tower
                    enemy_tower = organ
                elif organ_subtype == "ACTOR_SUB_CRYSTAL":  # 24 is ACTOR_SUB_CRYSTAL, base crystal
                    enemy_spring = organ

        for reward_name, reward_struct in cul_calc_frame_map.items():
            reward_struct.last_frame_value = reward_struct.cur_frame_value
            # Money
            # 金钱
            if reward_name == "money":
                if main_hero['moneyCnt'] >= get_hero_money_limit(main_hero['actor_state']['config_id']):
                    reward_struct.cur_frame_value = reward_struct.last_frame_value - 1
                else:
                    reward_struct.cur_frame_value = main_hero["moneyCnt"]
            # Health points
            # 生命值
            elif reward_name == "hp_point":
                reward_struct.cur_frame_value = math.sqrt(math.sqrt(1.0 * main_hero_hp / main_hero_max_hp))
            # Energy points
            # 法力值
            elif reward_name == "ep_rate":
                if main_hero_max_ep == 0 or main_hero_hp <= 0:
                    reward_struct.cur_frame_value = 0
                else:
                    reward_struct.cur_frame_value = main_hero_ep / float(main_hero_max_ep)
            # Kills
            # 击杀
            elif reward_name == "kill":
                reward_struct.cur_frame_value = main_hero["killCnt"]
            # Deaths
            # 死亡
            elif reward_name == "death":
                reward_struct.cur_frame_value = main_hero["deadCnt"]
            # Tower health points
            # 塔血量
            elif reward_name == "tower_hp_point":
                reward_struct.cur_frame_value = 1.0 * main_tower["hp"] / main_tower["max_hp"]
            # Last hit
            # 补刀
            elif reward_name == "last_hit":
                reward_struct.cur_frame_value = 0.0
                frame_action = frame_data["frame_action"]
                if "dead_action" in frame_action:
                    dead_actions = frame_action["dead_action"]
                    for dead_action in dead_actions:
                        if (
                            dead_action["killer"]["runtime_id"] == main_hero["actor_state"]["runtime_id"]
                            and dead_action["death"]["sub_type"] == "ACTOR_SUB_SOLDIER"
                        ):
                            reward_struct.cur_frame_value += 1.0
                        elif (
                            dead_action["killer"]["runtime_id"] == enemy_hero["actor_state"]["runtime_id"]
                            and dead_action["death"]["sub_type"] == "ACTOR_SUB_SOLDIER"
                        ):
                            reward_struct.cur_frame_value -= 1.0
            # Experience points
            # 经验值
            elif reward_name == "exp":
                reward_struct.cur_frame_value = self.calculate_exp_sum(main_hero)
            # Forward
            # 前进
            elif reward_name == "forward":
                reward_struct.cur_frame_value = self.calculate_forward(main_hero, main_tower, enemy_tower)
            # Close to cake
            # 靠近血包
            elif reward_name == "close_to_cake":
                reward_struct.cur_frame_value = self.calculate_close_to_cake(main_hero, frame_data)
            # Moving speed
            # 移速
            elif reward_name == "extra_mov_spd":
                reward_struct.cur_frame_value = self.calculate_extra_mov_spd(main_hero, frame_data)
            # Kill monster
            # 击杀野怪
            elif reward_name == "kill_monster":
                reward_struct.cur_frame_value = self.calculate_kill_monster(main_hero, frame_data)
            # Hit target
            elif reward_name == "hit_target":
                reward_struct.cur_frame_value = self.calculate_hit_target(main_hero, enemy_hero, frame_data)
            # Kiting
            elif reward_name == "kiting":
                reward_struct.cur_frame_value = self.calculate_kiting(main_hero, enemy_hero)
            # Hero unique
            elif reward_name == "hit_by_organ":
                reward_struct.cur_frame_value = self.calculate_hit_by_organ(main_hero, frame_data)

    def calculate_hit_by_organ(self, main_hero, frame_data):
        if main_hero["actor_state"]['location']['y'] == 100000:
            return 0
        hit_by_organ_reward = 0.0
        main_hero_camp = main_hero['actor_state']['camp'] 
        npc_list = frame_data["npc_states"]
        # print(frame_data["frameNo"])
        # print(main_hero["actor_state"]['location'])
        for npc_ in npc_list:
            if npc_['config_id'] in [46, 44] and npc_['camp'] != main_hero_camp and npc_['attack_target'] == main_hero["actor_state"]["runtime_id"]:
                hit_by_organ_reward += 0.4
            if npc_['config_id'] in  [1111,1112] and npc_['camp'] != main_hero_camp and npc_['attack_target'] == main_hero["actor_state"]["runtime_id"]:
                hit_by_organ_reward += 0.4
        return hit_by_organ_reward


    # Kiting
    def calculate_kiting(self, main_hero, enemy_hero):
        main_hero_atk_range = main_hero["actor_state"]["attack_range"]
        if main_hero["actor_state"]["config_id"] == 508 and main_hero["level"] > 1:
            main_hero_atk_range = 9500
        
        hero_dist = main_hero_atk_range
        if enemy_hero["actor_state"]["forward"]["x"] != 100000:
            main_hero_pos = (
                main_hero["actor_state"]["location"]["x"],
                main_hero["actor_state"]["location"]["z"],
            )
            enemy_hero_pos = (
                enemy_hero["actor_state"]["location"]["x"],
                enemy_hero["actor_state"]["location"]["z"],
            )
            hero_dist = min(math.dist(main_hero_pos, enemy_hero_pos), main_hero_atk_range)
            
            # 根据敌方英雄状态调整 kiting 奖励
            enemy_hp_ratio = enemy_hero["actor_state"]["hp"] / enemy_hero["actor_state"]["max_hp"]
            self_hp_ratio = main_hero["actor_state"]["hp"] / main_hero["actor_state"]["max_hp"]
            
            # 当我方血量低或敌方血量高时，增加保持距离的奖励
            kiting_multiplier = 1.0
            if self_hp_ratio < 0.4 or enemy_hp_ratio > 0.7:
                kiting_multiplier = 2.0
            
            kiting_reward = ((hero_dist - main_hero_atk_range) / main_hero_atk_range) * kiting_multiplier
            
            if main_hero["actor_state"]["config_id"] == 199:  # 公孙离
                kiting_reward *= 3.5  # 增加公孙离的 kiting 奖励
            elif main_hero["actor_state"]["config_id"] == 508:  # 伽罗
                kiting_reward *= 4.0  # 增加伽罗的 kiting 奖励
            
            return kiting_reward
        return 0


    # Hit target
    def calculate_hit_target(self, main_hero, enemy_hero, frame_data):
        main_hero_id = main_hero["actor_state"]["config_id"]
        hit_reward = 0.0
        hurt_reward = 0.0
        hero_coef = 1.5  # 提高命中英雄的奖励系数
        other_coef = 0.05  # 降低命中其他目标的奖励
        
        # 缓存常用值
        enemy_runtime_id = enemy_hero["actor_state"]["runtime_id"]
        main_hero_runtime_id = main_hero["actor_state"]["runtime_id"]
        
        if "hit_target_info" in main_hero["actor_state"]:
            for item in main_hero["actor_state"]["hit_target_info"]:
                hit_target = item["hit_target"]
                skill_id = item["skill_id"]
                skill_hero_id = skill_id // 100
                
                if skill_hero_id != main_hero_id:
                    continue
                    
                skill_no = (skill_id // 10) % 10
                weight = get_skill_reward_weight(main_hero_id, skill_no)
                
                if hit_target == enemy_runtime_id:
                    # 根据英雄血量比例调整奖励
                    enemy_hp_ratio = enemy_hero["actor_state"]["hp"] / enemy_hero["actor_state"]["max_hp"]
                    hp_bonus = max(0.5, 1.0 - enemy_hp_ratio)  # 敌人血量越低，奖励越高
                    
                    if main_hero_id == 508 and skill_no == 0 and main_hero["actor_state"]["attack_range"] == 9500:
                        hit_reward += weight * hero_coef * 10 * hp_bonus  # 提高伽罗远程攻击奖励
                    else:
                        hit_reward += weight * hero_coef * hp_bonus
                else:
                    hit_reward += weight * other_coef

        # 优化受伤惩罚
        if "takeHurtInfos" in main_hero:
            take_hurt_info = main_hero["takeHurtInfos"]
            for item in take_hurt_info:
                attacker = item["atker"]
                skill_id = item["skillSlot"]
                source_type = item["sourceType"]
                if source_type == "SKILL_USE_FROM_TYPE_SKILL" and skill_id <= 3:
                    if attacker == enemy_runtime_id:
                        # 根据自身血量比例调整惩罚
                        self_hp_ratio = main_hero["actor_state"]["hp"] / main_hero["actor_state"]["max_hp"]
                        hp_penalty = 1.0 + max(0, 0.5 - self_hp_ratio) * 2  # 自身血量越低，惩罚越重
                        hurt_reward += get_skill_reward_weight(main_hero_id, skill_id) * hero_coef * hp_penalty
                    else:
                        hurt_reward += get_skill_reward_weight(main_hero_id, skill_id) * other_coef
        
        return hit_reward - hurt_reward

    # Kill monster
    def calculate_kill_monster(self, main_hero, frame_data):
        monster_info = [npc for npc in frame_data["npc_states"] if npc.get('camp') == "PLAYERCAMP_MID"]
        frame_action = frame_data["frame_action"]
        if "dead_action" in frame_action:
            dead_actions = frame_action["dead_action"]
            for dead_action in dead_actions:
                if dead_action["death"]["camp"] == "PLAYERCAMP_MID" and dead_action["killer"]["runtime_id"] == main_hero["actor_state"]["runtime_id"]:
                    return 1.0
        return 0
        

    # # Moving speed
    def calculate_extra_mov_spd(self, main_hero, frame_data):
        # 移速信息，根据英雄、装备计算
        # 算main_hero
        cur_mov_spd = main_hero['actor_state']['values']['mov_spd']
        if cur_mov_spd == 0:
            return 0
        main_hero_equips = main_hero['equip_state']['equips']
        main_hero_id = main_hero['actor_state']['config_id']
        # 判断main_hero是什么英雄
        main_hero_equips_id = []
        for equip in main_hero_equips:
            main_hero_equips_id.append(equip['configId'])
        basic_mov_speed = 0
        if main_hero_id == 508:
            # 如果是伽罗，判断是长还是短箭
            if main_hero['actor_state']['attack_range'] == 9500:
                long_or_short = 0
            else:
                long_or_short = 1
            
            # 如果是伽罗，判断他的装备到哪个阶段了
            if 1136 in main_hero_equips_id:
                # 如果到最后的影刃阶段
                basic_mov_speed = [4879, 5131][long_or_short]
            elif 1135 in main_hero_equips_id:
                # 如果闪电匕首阶段
                basic_mov_speed = [4879, 5112][long_or_short]
            elif 1123 in main_hero_equips_id:
                # 如果狂暴双刃阶段
                basic_mov_speed = [4551, 4795][long_or_short]
            elif 1425 in main_hero_equips_id:
                # 如果攻速鞋阶段
                basic_mov_speed = [4346, 4558][long_or_short]
            elif 1411 in main_hero_equips_id:
                # 如果草鞋阶段
                basic_mov_speed = 4028
            else:
                basic_mov_speed = 3710
        elif main_hero_id == 133:
            if 1421 in main_hero_equips_id:
                # 如果到布甲鞋阶段
                basic_mov_speed = 4452
            elif 1411 in main_hero_equips_id:
                # 如果草鞋阶段
                basic_mov_speed = 4134
            else:
                basic_mov_speed = 3816
        elif main_hero_id == 199:
            if 1421 in main_hero_equips_id:
                # 如果到布甲鞋阶段
                basic_mov_speed = 4311
            elif 1411 in main_hero_equips_id:
                # 如果草鞋阶段
                basic_mov_speed = 3993
            else:
                basic_mov_speed = 3675  # 加速后4200
        extr_mov_spd = cur_mov_spd - basic_mov_speed
        if main_hero_id == 199:
            extr_mov_spd *= 1
        elif main_hero_id == 508:
            extr_mov_spd *= 1
        return extr_mov_spd * 0.0001

    # Close to cake
    def calculate_close_to_cake(self, main_hero, frame_data):
        hero_pos = (
            main_hero["actor_state"]["location"]["x"],
            main_hero["actor_state"]["location"]["z"],
        )
        hero_forward = (
            main_hero["actor_state"]["forward"]["x"],
            main_hero["actor_state"]["forward"]["z"],
        )
        proj_len = 0.0
        cake_1_pos = (-15220, -15120)
        cake_2_pos = (15340, 15100)
        if "cakes" in frame_data.keys():
            if main_hero["actor_state"]["camp"] == "PLAYERCAMP_1":
                direction_1 = (cake_1_pos[0] - hero_pos[0], cake_1_pos[1] - hero_pos[1])
                proj_len = (direction_1[0] * hero_forward[0] + direction_1[1] * hero_forward[1] + 1e-5) / math.sqrt(direction_1[0]**2 + direction_1[1]**2 + 1e-5)
            elif main_hero["actor_state"]["camp"] == "PLAYERCAMP_2":
                direction_2 = (cake_2_pos[0] - hero_pos[0], cake_2_pos[1] - hero_pos[1])
                proj_len = (direction_2[0] * hero_forward[0] + direction_2[1] * hero_forward[1] + 1e-5) / math.sqrt(direction_2[0]**2 + direction_2[1]**2 + 1e-5)
        proj_len *= (main_hero["actor_state"]["max_hp"] - main_hero["actor_state"]["hp"]) / main_hero["actor_state"]["max_hp"]
        return proj_len/1000
        

    # Calculate the total amount of experience gained using agent level and current experience value
    # 用智能体等级和当前经验值，计算获得经验值的总量
    def calculate_exp_sum(self, this_hero_info):
        exp_sum = 0.0
        for i in range(1, this_hero_info["level"]):
            exp_sum += self.m_each_level_max_exp[i]
        exp_sum += this_hero_info["exp"]
        return exp_sum

    # Calculate the forward reward based on the distance between the agent and both defensive towers
    # 用智能体到双方防御塔的距离，计算前进奖励
    def calculate_forward(self, main_hero, main_tower, enemy_tower):
        # 提前计算常用值
        hero_hp_ratio = main_hero["actor_state"]["hp"] / main_hero["actor_state"]["max_hp"]
        
        main_tower_pos = (main_tower["location"]["x"], main_tower["location"]["z"])
        enemy_tower_pos = (enemy_tower["location"]["x"], enemy_tower["location"]["z"])
        hero_pos = (
            main_hero["actor_state"]["location"]["x"],
            main_hero["actor_state"]["location"]["z"],
        )
        
        dist_hero2emy = math.dist(hero_pos, enemy_tower_pos)
        dist_main2emy = math.dist(main_tower_pos, enemy_tower_pos)
        
        # 根据血量和等级调整前进奖励
        forward_multiplier = 1.0
        if hero_hp_ratio > 0.8 and main_hero["level"] >= 4:  # 血量充足且等级达到要求时
            forward_multiplier = 1.5
        elif hero_hp_ratio < 0.3:  # 血量过低时降低前进奖励
            forward_multiplier = 0.3
        
        if dist_hero2emy > dist_main2emy:
            forward_value = ((dist_main2emy - dist_hero2emy) / dist_main2emy) * forward_multiplier
            return forward_value
        return 0

    # Calculate the reward item information for both sides using frame data
    # 用帧数据来计算两边的奖励子项信息
    def frame_data_process(self, frame_data):
        main_camp, enemy_camp = -1, -1

        for hero in frame_data["hero_states"]:
            if hero["player_id"] == self.main_hero_player_id:
                main_camp = hero["actor_state"]["camp"]
                self.main_hero_camp = main_camp
            else:
                enemy_camp = hero["actor_state"]["camp"]
        self.set_cur_calc_frame_vec(self.m_main_calc_frame_map, frame_data, main_camp)
        self.set_cur_calc_frame_vec(self.m_enemy_calc_frame_map, frame_data, enemy_camp)

    # Use the values obtained in each frame to calculate the corresponding reward value
    # 用每一帧得到的奖励子项信息来计算对应的奖励值
    def get_reward(self, frame_data, reward_dict):
        if self.start_time == -1:
            self.start_time = time.time()
        # this_time = 999
        # delta_time = int(this_time - self.start_time)
        # self.time_step = min(int(delta_time / (rewardChangeInterval / rewardChangeStep)), rewardChangeStep-1)
        timeNode = 7200
        
        # print(delta_time, self.time_step)
        frame_no = frame_data['frameNo']
        stageNo = 0
        if frame_no <= 3600:
            stageNo = 1
        elif frame_no <= 12000:
            stageNo = 2
        else:
            stageNo = 3
        # if delta_time >= 7200:
        #     timeNode = 7200
        origin_map = change_calc_frame_map(self.m_cur_calc_frame_map, stageNo, frame_no)
        self.m_cur_calc_frame_map = change_target_calc_frame_map(self.m_cur_calc_frame_map, stageNo, frame_no)
        self.m_cur_calc_frame_map = get_progressive_reward_map(origin_map, self.m_cur_calc_frame_map,  timeNode)
        # print(self.m_cur_calc_frame_map["money"].weight)
        reward_dict.clear()
        reward_sum, weight_sum = 0.0, 0.0

        # 调整各项奖励的权重
        reward_weights = {
            "hp_point": 1.2,      # 提高生命值权重
            "money": 0.8,         # 降低金钱权重
            "kill": 1.5,          # 提高击杀权重
            "death": 2.0,         # 提高死亡惩罚
            "tower_hp_point": 1.3, # 提高防御塔血量权重
            "hit_target": 1.4,    # 提高命中目标权重
            "kiting": 1.3,        # 提高 kiting 权重
            "forward": 0.9,       # 降低前进权重
        }
        
        for reward_name, reward_struct in self.m_cur_calc_frame_map.items():
            if reward_name in reward_weights:
                reward_struct.weight *= reward_weights[reward_name]
            
        for reward_name, reward_struct in self.m_cur_calc_frame_map.items():
            if reward_name == "hp_point":
                if (
                    self.m_main_calc_frame_map[reward_name].last_frame_value == 0.0
                    and self.m_enemy_calc_frame_map[reward_name].last_frame_value == 0.0
                ):
                    reward_struct.cur_frame_value = 0
                    reward_struct.last_frame_value = 0
                elif self.m_main_calc_frame_map[reward_name].last_frame_value == 0.0:
                    reward_struct.cur_frame_value = 0 - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                    reward_struct.last_frame_value = 0 - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                elif self.m_enemy_calc_frame_map[reward_name].last_frame_value == 0.0:
                    reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value - 0
                    reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value - 0
                else:
                    reward_struct.cur_frame_value = (
                        self.m_main_calc_frame_map[reward_name].cur_frame_value
                        - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                    )
                    reward_struct.last_frame_value = (
                        self.m_main_calc_frame_map[reward_name].last_frame_value
                        - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                    )
                reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value
            elif reward_name == "ep_rate":
                reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value
                reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value
                if reward_struct.last_frame_value > 0:
                    reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value
                else:
                    reward_struct.value = 0
            elif reward_name == "exp":
                main_hero = None
                for hero in frame_data["hero_states"]:
                    if hero["player_id"] == self.main_hero_player_id:
                        main_hero = hero
                if main_hero and main_hero["level"] >= 15:
                    reward_struct.value = 0
                else:
                    reward_struct.cur_frame_value = (
                        self.m_main_calc_frame_map[reward_name].cur_frame_value
                        - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                    )
                    reward_struct.last_frame_value = (
                        self.m_main_calc_frame_map[reward_name].last_frame_value
                        - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                    )
                    reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value
            elif reward_name == "forward":
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
            elif reward_name == "last_hit":
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
            elif reward_name == "close_to_cake":
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
            elif reward_name == "extra_mov_spd":
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
            elif reward_name == "kill_monster":
                reward_struct.value = (self.m_main_calc_frame_map[reward_name].cur_frame_value
                                      - self.m_enemy_calc_frame_map[reward_name].cur_frame_value)
            elif reward_name == "hit_target":
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
            elif reward_name == "kiting":
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
            elif reward_name == "hit_by_organ":
                reward_struct.value = (self.m_main_calc_frame_map[reward_name].cur_frame_value
                                      - 0.5 * self.m_enemy_calc_frame_map[reward_name].cur_frame_value)
            else:
                reward_struct.cur_frame_value = (
                    self.m_main_calc_frame_map[reward_name].cur_frame_value
                    - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                )
                reward_struct.last_frame_value = (
                    self.m_main_calc_frame_map[reward_name].last_frame_value
                    - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                )
                reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value

            weight_sum += reward_struct.weight
            reward_sum += reward_struct.value * reward_struct.weight
            reward_dict[reward_name] = reward_struct.value
        reward_dict["reward_sum"] = reward_sum

class ActorState(TypedDict):
    camp: str
    hp: int
    max_hp: int
    runtime_id: int
    config_id: int
    location: Dict[str, float]
    forward: Dict[str, float]
    attack_range: int
    values: Dict[str, float]

class HeroState(TypedDict):
    actor_state: ActorState
    level: int
    exp: int
    moneyCnt: int
    killCnt: int
    deadCnt: int
