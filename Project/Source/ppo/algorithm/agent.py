#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2024 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import os
from ppo.model.model import Model
from ppo.feature.definition import *
import numpy as np
from kaiwu_agent.agent.base_agent import (
    BaseAgent,
    predict_wrapper,
    exploit_wrapper,
    learn_wrapper,
    save_model_wrapper,
    load_model_wrapper,
)

from ppo.config import Config
from kaiwu_agent.utils.common_func import attached
from ppo.feature.reward_manager import GameRewardManager

buff_info_len = 222

buff_id_map = {
    10000: 0, 
    10010: 1, 
    10014: 2, 
    11001: 3, 
    11002: 4, 
    50000: 5, 
    50001: 6, 
    90015: 7, 
    90019: 8, 
    90025: 9, 
    131956: 10, 
    133000: 11, 
    133001: 12, 
    133010: 13, 
    133011: 14, 
    133020: 15, 
    133090: 16, 
    133100: 17, 
    133200: 18, 
    133300: 19, 
    133310: 20, 
    133320: 21, 
    133330: 22, 
    133350: 23, 
    133390: 24, 
    133950: 25, 
    133951: 26, 
    199000: 27, 
    199001: 28, 
    199050: 29, 
    199060: 30, 
    199070: 31, 
    199150: 32, 
    199160: 33, 
    199200: 34, 
    199300: 35, 
    199301: 36, 
    199390: 37, 
    199391: 38, 
    199900: 39, 
    199910: 40, 
    199970: 41, 
    500009: 42, 
    508000: 43, 
    508001: 44, 
    508010: 45, 
    508020: 46, 
    508100: 47, 
    508140: 48, 
    508141: 49, 
    508150: 50, 
    508151: 51, 
    508152: 52, 
    508160: 53, 
    508161: 54, 
    508162: 55, 
    508170: 56, 
    508180: 57, 
    508200: 58, 
    508201: 59, 
    508210: 60, 
    508211: 61, 
    508290: 62, 
    508291: 63, 
    508300: 64, 
    508310: 65, 
    508350: 66, 
    508351: 67, 
    508360: 68, 
    508391: 69, 
    508392: 70, 
    911260: 71, 
    911261: 72, 
    911272: 73, 
    911273: 74, 
    911274: 75, 
    911275: 76, 
    911276: 77, 
    911290: 78, 
    911330: 79, 
    911332: 80, 
    911340: 81, 
    911341: 82, 
    911342: 83, 
    911350: 84, 
    911351: 85, 
    911352: 86, 
    911354: 87, 
    911355: 88, 
    911356: 89, 
    911357: 90, 
    911358: 91, 
    911359: 92, 
    911360: 93, 
    911361: 94, 
    911541: 95, 
    911551: 96, 
    911553: 97, 
    911580: 98, 
    911581: 99, 
    912230: 100,
    912231: 101, 
    912331: 102, 
    912332: 103, 
    913370: 104, 
    913375: 105, 
    914110: 106, 
    914210: 107, 
    914211: 108, 
    914250: 109, 
    919902: 110
}


@attached
class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        self.extra_set = set()
        self.cur_model_name = ""
        self.device = device
        # Create Model and convert the model to achannel-last memory format to achieve better performance.
        # 创建模型, 将模型转换为通道后内存格式，以获得更好的性能。
        self.model = Model()
        self.model = self.model.to(self.device).to(memory_format=torch.channels_last)
        
        self.model.set_eval_mode()
        example_inputs = [torch.rand(1, 725+114+buff_info_len).to(self.device), torch.rand(1, 256).to(self.device), torch.rand(1, 256).to(self.device)]
        self.scripted_model = torch.jit.trace(self.model, example_inputs=(example_inputs,)).to(self.device)


        # config info
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.lstm_real_size = Config.LSTM_REAL_SIZE
        self.lstm_hidden = np.zeros([self.lstm_real_size])
        self.lstm_cell = np.zeros([self.lstm_real_size])
        self.label_size_list = Config.LABEL_SIZE_LIST
        self.legal_action_size = Config.LEGAL_ACTION_SIZE_LIST
        self.seri_vec_split_shape = Config.SERI_VEC_SPLIT_SHAPE
        self.data_split_shape = Config.DATA_SPLIT_SHAPE
        self.cut_points = [value[0] for value in Config.data_shapes]

        # env info
        self.hero_camp = 0
        self.player_id = 0
        self.game_id = None

        # learning info
        self.train_step = 0
        initial_lr = Config.INIT_LEARNING_RATE_START
        parameters = self.model.parameters()
        self.optimizer = torch.optim.Adam(params=parameters, lr=initial_lr, betas=(0.9, 0.999), eps=1e-8)
        self.parameters = [p for param_group in self.optimizer.param_groups for p in param_group["params"]]

        # tools
        self.reward_manager = None
        self.logger = logger
        self.monitor = monitor

        super().__init__(agent_type, device, logger, monitor)


    def _model_inference(self, list_obs_data):
        # 使用网络进行推理
        # Using the network for inference
        feature = [obs_data.feature for obs_data in list_obs_data]
        legal_action = [obs_data.legal_action for obs_data in list_obs_data]
        lstm_cell = [obs_data.lstm_cell for obs_data in list_obs_data]
        lstm_hidden = [obs_data.lstm_hidden for obs_data in list_obs_data]

        input_list = [np.array(feature), np.array(lstm_cell), np.array(lstm_hidden)]
        torch_inputs = [torch.from_numpy(nparr).to(torch.float32) for nparr in input_list]
        for i, data in enumerate(torch_inputs):
            data = data.reshape(-1)
            torch_inputs[i] = data.float()

        feature, lstm_cell, lstm_hidden = torch_inputs
        feature_vec = feature.reshape(-1, self.seri_vec_split_shape[0][0])
        lstm_hidden_state = lstm_hidden.reshape(-1, self.lstm_real_size)
        lstm_cell_state = lstm_cell.reshape(-1, self.lstm_real_size)

        format_inputs = [feature_vec, lstm_hidden_state, lstm_cell_state]

        self.scripted_model.eval()
        with torch.no_grad():
            output_list = self.scripted_model(format_inputs)

        np_output = []
        cnt = 0
        for output in output_list:
            if cnt == 1:
                np_output.append(output)
            else:
                np_output.append(output.numpy())
            cnt += 1

        logits, value, _lstm_cell, _lstm_hidden = np_output[:4]

        _lstm_cell = _lstm_cell.squeeze(axis=0)
        _lstm_hidden = _lstm_hidden.squeeze(axis=0)

        list_act_data = list()
        for i in range(len(legal_action)):
            prob, action, d_action = self._sample_masked_action(logits[i], legal_action[i])
            list_act_data.append(
                ActData(
                    action=action,
                    d_action=d_action,
                    prob=prob,
                    value=value,
                    lstm_cell=_lstm_cell[i],
                    lstm_hidden=_lstm_hidden[i],
                )
            )
        return list_act_data

    @predict_wrapper
    def predict(self, list_obs_data):
        return self._model_inference(list_obs_data)

    @exploit_wrapper
    def exploit(self, state_dict):
        # Evaluation task will not call agent.reset in the workflow. Users can use the game_id to determine whether a new environment
        # 评估任务不会在workflow中重置agent，用户可以通过game_id判断是否是新的对局，并根据新对局对agent进行重置
        game_id = state_dict["game_id"]
        if self.game_id != game_id:
            player_id = state_dict["player_id"]
            camp = state_dict["player_camp"]
            self.reset(camp, player_id)
            self.game_id = game_id

        # exploit is automatically called when submitting an evaluation task.
        # The parameter is the state_dict returned by env, and it returns the action used by env.step.
        # exploit在提交评估任务时自动调用，参数为env返回的state_dict, 返回env.step使用的action
        obs_data = self.observation_process(state_dict)
        # Call _model_inference for model inference, executing local model inference
        # 模型推理调用_model_inference, 执行本地模型推理
        act_data = self._model_inference([obs_data])[0]
        self.update_status(obs_data, act_data)
        return self.action_process(state_dict, act_data, False)

    def train_predict(self, state_dict):
        obs_data = self.observation_process(state_dict)
        # Call agent.predict for distributed model inference
        # 调用agent.predict，执行分布式模型推理
        act_data = self.predict([obs_data])[0]
        self.update_status(obs_data, act_data)
        return self.action_process(state_dict, act_data, True)

    def eval_predict(self, state_dict):
        obs_data = self.observation_process(state_dict)
        # Call agent.predict for distributed model inference
        # 调用agent.predict，执行分布式模型推理
        act_data = self.predict([obs_data])[0]
        self.update_status(obs_data, act_data)
        return self.action_process(state_dict, act_data, False)

    def action_process(self, state_dict, act_data, is_stochastic):
        if is_stochastic:
            # Use stochastic sampling action
            # 采用随机采样动作 action
            return act_data.action
        else:
            # Use the action with the highest probability
            # 采用最大概率动作 d_action
            return act_data.d_action

    def observation_process(self, state_dict):
        feature_vec, legal_action = (
            state_dict["observation"],
            state_dict["legal_action"],
        )

        frame_state = state_dict["frame_state"]
        hero_list = frame_state["hero_states"]
        frame_no = frame_state["frameNo"]
        npc_list = frame_state["npc_states"]
        main_hero_player_id = state_dict['player_id']
        main_hero = None
        enemy_hero = None
        for hero in hero_list:
            hero_camp = hero["actor_state"]["camp"]
            hero_hp = hero["actor_state"]["hp"]
            hero_player_id = hero["player_id"]
            if hero_player_id == main_hero_player_id:
                main_hero = hero
            else:
                enemy_hero = hero

        # 添加特征 84 = 6 + 14 + 2 + 18 + buff_info_len
        # 英雄ID向量 6
        CONFIG_ID_MAP = {
            133: 0,
            199: 1,
            508: 2,
        }
        main_hero_id = main_hero["actor_state"]["config_id"]
        enemy_hero_id = enemy_hero["actor_state"]["config_id"]
        hero_id_one_hot = [0] * 6
        hero_id_one_hot[CONFIG_ID_MAP[main_hero_id]] = 1
        hero_id_one_hot[CONFIG_ID_MAP[enemy_hero_id]+3] = 1
        hero_id_one_hot = np.array(hero_id_one_hot)
        feature_vec[705:711] = hero_id_one_hot

        # 野怪信息 14
        monster_feature = [0] * 14
        monster = [npc for npc in npc_list if npc.get('camp') == "PLAYERCAMP_MID"]
        r = 1.0
        if main_hero['actor_state']['camp'] == "PLAYERCAMP_2":
            r = -1.0
        if len(monster) >= 1:
            monster0 = monster[0]
            monster_feature[0] = monster0['location']['x']/10000*r
            monster_feature[1] = monster0['location']['z']/10000*r
            monster_feature[2] = monster0['forward']['x']/1000*r
            monster_feature[3] = monster0['forward']['z']/1000*r
            monster_feature[4] = monster0['hp']/monster0['max_hp']
            monster_feature[5] = monster0['values']['phy_atk']/1000
            monster_feature[6] = monster0['values']['phy_def']/1000
            monster_feature[7] = monster0['values']['mgc_atk']/1000
            monster_feature[8] = monster0['values']['mgc_def']/1000
            monster_feature[9] = monster0['values']['mov_spd']/10000
            monster_feature[10] = monster0['attack_range']/1000
            monster_feature[11] = monster0['attack_target']/1000
            # behav_mode
            monster_feature[12] = monster0['kill_income']/100
            monster_feature[13] = monster0['sight_area']/10000
        feature_vec[711:725] =  monster_feature

        # 血包信息 2
        cake_feature = [0] * 2
        main_cake = 0
        emy_cake = 0
        if "cakes" in frame_state.keys():
            cake_list = frame_state['cakes']
            for cake in cake_list:
                if (cake['collider']['location']['x'] < 0):
                    if main_hero['actor_state']['camp'] == "PLAYERCAMP_1":
                        main_cake = 1
                    else:
                        emy_cake = 1
                else:
                    if main_hero['actor_state']['camp'] == "PLAYERCAMP_1":
                        emy_cake = 1
                    else:
                        main_cake = 1
        cake_feature[0] = main_cake
        cake_feature[1] = emy_cake

        # 子弹信息 30
        bullet_feature = [0] * 3 * 5 * 2
        # global bullet_type_set
        # global bullet_id_set
        main_runtime_id = main_hero['actor_state']['runtime_id']
        enemy_runtime_id = enemy_hero['actor_state']['runtime_id']
        
        main_hero_camp = main_hero['actor_state']['camp']
        enemy_hero_camp = enemy_hero['actor_state']['camp']
        main_first_tower_id = 0
        main_second_tower_id = 0
        main_first_tower_runtime_id = 0
        main_second_tower_runtime_id = 0
        enemy_first_tower_id = 0
        enemy_second_tower_id = 0
        enemy_first_tower_runtime_id = 0
        enemy_second_tower_runtime_id = 0
        for npc in npc_list:
            # 找到己方一二塔的id和runtime id
            # 如果npc是一塔
            if npc['config_id'] in [44,46]:
                if npc['camp'] == main_hero_camp:
                    main_first_tower_id = npc['config_id']
                    main_first_tower_runtime_id = npc['runtime_id']
                else:
                    enemy_first_tower_id = npc['config_id']
                    enemy_first_tower_runtime_id = npc['runtime_id']
            elif npc['config_id'] in [1111,1112]:
                if npc['camp'] == main_hero_camp:
                    main_second_tower_id = npc['config_id']
                    main_second_tower_runtime_id = npc['runtime_id']
                else:
                    enemy_second_tower_id = npc['config_id']
                    enemy_second_tower_runtime_id = npc['runtime_id']
            
        # 整理npc信息
        # 小兵：8800 50 塔   1500 75 兵(6800 6803)    7000 46 兵(6801 6804)   13000 150 塔（46，44）  12000  0 水晶  12000 50 塔（1111 1112）      
        if "bullets" in frame_state.keys():
            bullet_list = frame_state["bullets"]
            main_hero_bullets = [bullet for bullet in bullet_list if bullet.get('source_actor') == main_runtime_id]
            main_slot_skill_0 = []
            main_slot_skill_1 = []
            main_slot_skill_2 = []
            main_first_tow = []
            main_second_tow = []
            r = 1.0
            if main_hero['actor_state']['camp'] == "PLAYERCAMP_2":
                r = -1.0
            for bullet in main_hero_bullets:
                if bullet['slot_type'] == 'SLOT_SKILL_0':
                    if len(main_slot_skill_0) == 0:
                        main_slot_skill_0 = [bullet['location']['x']/30000*r, bullet['location']['z']/30000*r, 1]
                    else:
                        main_slot_skill_0[2] += 1
                elif bullet['slot_type'] == 'SLOT_SKILL_1':
                    if len(main_slot_skill_1) == 0:
                        main_slot_skill_1 = [bullet['location']['x']/30000*r, bullet['location']['z']/30000*r, 1]
                    else:
                        main_slot_skill_1[2] += 1
                elif bullet['slot_type'] == 'SLOT_SKILL_2':
                    if len(main_slot_skill_2) == 0:
                        main_slot_skill_2 = [bullet['location']['x']/30000*r, bullet['location']['z']/30000*r, 1]
                    else:
                        main_slot_skill_2[2] += 1
                
            if len(main_slot_skill_0) > 0:
                bullet_feature[0:3] = main_slot_skill_0
            if len(main_slot_skill_1) > 0:
                bullet_feature[3:6] = main_slot_skill_1
            if len(main_slot_skill_2) > 0:
                bullet_feature[6:9] = main_slot_skill_2
            if len(main_first_tow) > 0:
                bullet_feature[9:12] = main_first_tow
            if len(main_second_tow) > 0:
                bullet_feature[12:15] = main_second_tow
            enemy_hero_bullets = [bullet for bullet in bullet_list if bullet.get('source_actor') == enemy_runtime_id]
            enemy_slot_skill_0 = []
            enemy_slot_skill_1 = []
            enemy_slot_skill_2 = []
            enemy_second_tow = []
            enemy_first_tow = []
            for bullet in enemy_hero_bullets:
                if bullet['slot_type'] == 'SLOT_SKILL_0':
                    if len(enemy_slot_skill_0) == 0:
                        enemy_slot_skill_0 = [bullet['location']['x']/30000*r, bullet['location']['z']/30000*r, 1]
                    else:
                        enemy_slot_skill_0[2] += 1
                elif bullet['slot_type'] == 'SLOT_SKILL_1':
                    if len(enemy_slot_skill_1) == 0:
                        enemy_slot_skill_1 = [bullet['location']['x']/30000*r, bullet['location']['z']/30000*r, 1]
                    else:
                        enemy_slot_skill_1[2] += 1
                elif bullet['slot_type'] == 'SLOT_SKILL_2':
                    if len(enemy_slot_skill_2) == 0:
                        enemy_slot_skill_2 = [bullet['location']['x']/30000*r, bullet['location']['z']/30000*r, 1]
                    else:
                        enemy_slot_skill_2[2] += 1
                

            for bullet in bullet_list:
                if bullet['source_actor'] == main_first_tower_runtime_id:
                    if len(main_first_tow) == 0:
                        main_first_tow = [bullet['location']['x']/30000*r, bullet['location']['z']/30000*r, 1]
                    else:
                        main_first_tow[2] += 1
                elif bullet['source_actor'] == main_second_tower_runtime_id:
                    if len(main_second_tow) == 0:
                        main_second_tow = [bullet['location']['x']/30000*r, bullet['location']['z']/30000*r, 1]
                    else:
                        main_second_tow[2] += 1
                elif bullet['source_actor'] == enemy_first_tower_runtime_id:
                    if len(enemy_first_tow) == 0:
                        enemy_first_tow = [bullet['location']['x']/30000*r, bullet['location']['z']/30000*r, 1]
                    else:
                        enemy_first_tow[2] += 1
                elif bullet['source_actor'] == enemy_second_tower_runtime_id:
                    if len(enemy_second_tow) == 0:
                        enemy_second_tow = [bullet['location']['x']/30000*r, bullet['location']['z']/30000*r, 1]
                    else:
                        enemy_second_tow[2] += 1
                
            if len(enemy_slot_skill_0) > 0:
                bullet_feature[9:12] = enemy_slot_skill_0
            if len(enemy_slot_skill_1) > 0:
                bullet_feature[12:15] = enemy_slot_skill_1
            if len(enemy_slot_skill_2) > 0:
                bullet_feature[15:18] = enemy_slot_skill_2
            if len(enemy_first_tow) > 0:
                bullet_feature[18:21] = enemy_first_tow
            if len(enemy_second_tow) > 0:
                bullet_feature[21:24] = enemy_second_tow

        # BUFF信息 buff_info_len
        buff_feature = [0] * buff_info_len
        if 'buff_skills' in main_hero['actor_state']['buff_state'].keys():
            for buff in main_hero['actor_state']['buff_state']['buff_skills']:
                if buff['configId'] not in buff_id_map.keys():
                    self.extra_set.add(buff['configId'])
                else:
                    buff_feature[buff_id_map[buff['configId']]] = 1
        if 'buff_skills' in enemy_hero['actor_state']['buff_state'].keys():
            for buff in enemy_hero['actor_state']['buff_state']['buff_skills']:
                if buff['configId'] not in buff_id_map.keys():
                    self.extra_set.add(buff['configId'])
                else:
                    buff_feature[buff_id_map[buff['configId']]+buff_info_len//2] = 1

        # 英雄和塔的target，英雄朝向，英雄动作 54 = 2 * 27
        # 英雄动作 COMMAND_TYPE_MoveDir: 向一个角度移动 move_dir[degree] = 135  如果是camp1就是-45 camp2 是135 所以角度也得镜像  
        # COMMAND_TYPE_MovePos 
        # COMMAND_TYPE_MoveStop 
        # COMMAND_TYPE_AttackCommon
        # COMMAND_TYPE_ObjSkill
        # COMMAND_TYPE_DirSkill
        ## COMMAND_TYPE_LearnSkill
        ## COMMAND_TYPE_BuyEquip

        #  133->狄仁杰，508->伽罗，199->公孙离
        # {'skillID': 90003, 'actorID': 13, 'slotType': 'SLOT_SKILL_4'}
        # 508 
        # {'skillID': 50810, 'actorID': 13, 'slotType': 'SLOT_SKILL_1'}
        # {'skillID': 50811, 'actorID': 13, 'slotType': 'SLOT_SKILL_1'}
        # {'skillID': 50830, 'actorID': 13, 'slotType': 'SLOT_SKILL_3'}
        # {'skillID': 50831, 'actorID': 13, 'slotType': 'SLOT_SKILL_3'}
        # 133
        # {'skillID': 13320, 'actorID': 16, 'slotType': 'SLOT_SKILL_2'}
        # 199
        # {'skillID': 19915, 'actorID': 16, 'slotType': 'SLOT_SKILL_1'}
        # {'skillID': 19920, 'actorID': 16, 'slotType': 'SLOT_SKILL_2'}
        # {'skillID': 19921, 'actorID': 16, 'slotType': 'SLOT_SKILL_2'} # 手上没闪的第一次2
        # {'skillID': 19925, 'actorID': 16, 'slotType': 'SLOT_SKILL_2'}
        # {'skillID': 19935, 'actorID': 16, 'slotType': 'SLOT_SKILL_3'}

        # {'skillID': 80115, 'actorID': 0, 'slotType': 'SLOT_SKILL_5', 'degree': 115}
        # 508
        # {'skillID': 50820, 'actorID': 0, 'slotType': 'SLOT_SKILL_2', 'degree': -120}
        # 133
        # {'skillID': 13310, 'actorID': 0, 'slotType': 'SLOT_SKILL_1', 'degree': 164}
        # {'skillID': 13330, 'actorID': 0, 'slotType': 'SLOT_SKILL_3', 'degree': -128}
        # 199
        # {'skillID': 19910, 'actorID': 0, 'slotType': 'SLOT_SKILL_1', 'degree': -107}
        # {'skillID': 19911, 'actorID': 0, 'slotType': 'SLOT_SKILL_1', 'degree': -51}
        # {'skillID': 19930, 'actorID': 0, 'slotType': 'SLOT_SKILL_3', 'degree': 135}
        # {'skillID': 19931, 'actorID': 0, 'slotType': 'SLOT_SKILL_3', 'degree': 128}


        # 一个英雄的动作 [no_action, mov_dir_degree, mov_pos_x, mov_pos_z, mov_stop, attack_command, attack_state), # 7
        # objskill:90003, 50810, 50811, 50830, 50831, 13320, 19915, 19920, 19921, 19925, 19935  # 11
        # dirskill:80115, 50820, 13310, 13330, 19910, 19911, 19930, 19931, degree]
        # attack_target晚点再说

        objskill_list = [90003, 50810, 50811, 50830, 50831, 13320, 19915, 19920, 19921, 19925, 19935]
        dirskill_list = [80115, 50820, 13310, 13330, 19910, 19911, 19930, 19931]

        r = 1.0
        if main_hero['actor_state']['camp'] == "PLAYERCAMP_2":
            r = -1.0
        main_hero_ation_list = None
        if 'real_cmd' not in main_hero:
            main_hero_ation_list = [1] + [0]*6 + [0]*11 + [0]*9
        else:
            main_hero_ation_list = [0]*7 + [0]*11 + [0]*9
            for com in main_hero['real_cmd']:
                if com['command_type'] == 'COMMAND_TYPE_MoveDir':
                    mov_degree = com['move_dir']['degree']
                    if r == -1:
                        mov_degree = (mov_degree + 180 - 360) if (mov_degree + 180 > 180) else (mov_degree + 180)
                    main_hero_ation_list[1] = mov_degree/180
                elif com['command_type'] == 'COMMAND_TYPE_MovePos':
                    x,z = com['move_pos']['destPos']['x'], com['move_pos']['destPos']['z']
                    main_hero_ation_list[2] = x/3000*r
                    main_hero_ation_list[3] = z/3000*r
                elif com['command_type'] == 'COMMAND_TYPE_MoveStop':
                    main_hero_ation_list[4] = 1
                elif com['command_type'] == 'COMMAND_TYPE_AttackCommon':
                    command_start = com['attack_common']['start']
                    main_hero_ation_list[5] = 1
                    main_hero_ation_list[6] = command_start
                elif com['command_type'] == 'COMMAND_TYPE_ObjSkill':
                    if com['obj_skill']['skillID'] in objskill_list:
                        skill_idx = objskill_list.index(com['obj_skill']['skillID'])
                        main_hero_ation_list[7+skill_idx] = 1
                elif com['command_type'] == 'COMMAND_TYPE_DirSkill':
                    if com['dir_skill']['skillID'] in dirskill_list:
                        skill_idx = dirskill_list.index(com['dir_skill']['skillID'])
                        skill_degree = com['dir_skill']['degree']
                        if r == -1:
                            skill_degree = (skill_degree + 180 - 360) if (skill_degree + 180 > 180) else (skill_degree + 180)
                        main_hero_ation_list[18+skill_idx] = 1
                        main_hero_ation_list[26] = skill_degree/180
        
        enemy_hero_ation_list = None
        if 'real_cmd' not in enemy_hero:
            enemy_hero_ation_list = [1] + [0]*6 + [0]*11 + [0]*9
        else:
            enemy_hero_ation_list = [0]*7 + [0]*11 + [0]*9
            for com in enemy_hero['real_cmd']:
                if com['command_type'] == 'COMMAND_TYPE_MoveDir':
                    mov_degree = com['move_dir']['degree']
                    if r == -1:
                        mov_degree = (mov_degree + 180 - 360) if (mov_degree + 180 > 180) else (mov_degree + 180)
                    enemy_hero_ation_list[1] = mov_degree/180
                elif com['command_type'] == 'COMMAND_TYPE_MovePos':
                    x,z = com['move_pos']['destPos']['x'], com['move_pos']['destPos']['z']
                    enemy_hero_ation_list[2] = x/3000*r
                    enemy_hero_ation_list[3] = z/3000*r
                elif com['command_type'] == 'COMMAND_TYPE_MoveStop':
                    enemy_hero_ation_list[4] = 1
                elif com['command_type'] == 'COMMAND_TYPE_AttackCommon':
                    command_start = com['attack_common']['start']
                    enemy_hero_ation_list[5] = 1
                    enemy_hero_ation_list[6] = command_start
                elif com['command_type'] == 'COMMAND_TYPE_ObjSkill':
                    if com['obj_skill']['skillID'] in objskill_list:
                        skill_idx = objskill_list.index(com['obj_skill']['skillID'])
                        enemy_hero_ation_list[7+skill_idx] = 1
                elif com['command_type'] == 'COMMAND_TYPE_DirSkill':
                    if com['dir_skill']['skillID'] in dirskill_list:
                        skill_idx = dirskill_list.index(com['dir_skill']['skillID'])
                        skill_degree = com['dir_skill']['degree']
                        if r == -1:
                            skill_degree = (skill_degree + 180 - 360) if (skill_degree + 180 > 180) else (skill_degree + 180)
                        enemy_hero_ation_list[18+skill_idx] = 1
                        enemy_hero_ation_list[26] = skill_degree/180

        # 维护一个runtime和config id对应关系
        main_hero_camp = main_hero['actor_state']['camp'] 
        runtime_config_id_dict = {}
        # 英雄
        for hero_ in hero_list:
            runtime_config_id_dict[hero_['actor_state']['runtime_id']] = hero_['actor_state']['config_id']
        # npc
        for npc_ in npc_list:
            runtime_config_id_dict[npc_['runtime_id']] = npc_['config_id']
            # 找到main的一二塔和 enemy的一二塔
            if npc_['config_id'] in  [46, 44]:
                if npc_['camp'] == main_hero_camp:
                    main_first_tow = npc_
                else:
                    enemy_first_tow = npc_
            if npc_['config_id'] in  [1111,1112]:
                if npc_['camp'] == main_hero_camp:
                    main_second_tow = npc_
                else:
                    enemy_second_tow = npc_


        # 判断attack的目标config id 28 = 8*2 + 3*4
        # 1500 75 近战兵(6800 1 6803 2)    7000 46 远程兵(6801 1 6804 2)   7000 129 跑车（6805 2 6802 1）   13000 150 外塔（46 2，44 1）   12000 50 塔（1111 1112）   野怪 6827   
        # [6803, 6804, 6827, 133, 1112, 6805]
        # [6800, 6801, 6827, 133, 6802]
        main_hero_attack_target = [0,0,0,0,0,0,0,0] # 近战兵，远程兵，跑车，外塔，内塔，野怪，英雄,阵亡单位
        if main_hero['actor_state']['attack_target'] != 0:
            try:
                attack_target_config_id = runtime_config_id_dict[main_hero['actor_state']['attack_target']]
                if attack_target_config_id in [6800, 6803]:
                    main_hero_attack_target[0] = 1
                elif attack_target_config_id in [6801, 6804]:
                    main_hero_attack_target[1] = 1
                elif attack_target_config_id in [6802, 6805]:
                    main_hero_attack_target[2] = 1
                elif attack_target_config_id in [46, 44]:
                    main_hero_attack_target[3] = 1
                elif attack_target_config_id in [1111, 1112]:
                    main_hero_attack_target[4] = 1
                elif attack_target_config_id in [6827]:
                    main_hero_attack_target[5] = 1
                elif attack_target_config_id in [133, 199, 508]:
                    main_hero_attack_target[6] = 1              
            except:
                # 攻击死了的单位
                main_hero_attack_target[7] = 1 
        enemy_hero_attack_target = [0,0,0,0,0,0,0,0] # 近战兵，远程兵，跑车，外塔，内塔，野怪，英雄,阵亡单位
        if enemy_hero['actor_state']['attack_target'] != 0:
            try:
                attack_target_config_id = runtime_config_id_dict[enemy_hero['actor_state']['attack_target']]
                if attack_target_config_id in [6800, 6803]:
                    enemy_hero_attack_target[0] = 1
                elif attack_target_config_id in [6801, 6804]:
                    enemy_hero_attack_target[1] = 1
                elif attack_target_config_id in [6802, 6805]:
                    enemy_hero_attack_target[2] = 1
                elif attack_target_config_id in [46, 44]:
                    enemy_hero_attack_target[3] = 1
                elif attack_target_config_id in [1111, 1112]:
                    enemy_hero_attack_target[4] = 1
                elif attack_target_config_id in [6827]:
                    enemy_hero_attack_target[5] = 1
                elif attack_target_config_id in [133, 199, 508]:
                    enemy_hero_attack_target[6] = 1              
            except:
                # 攻击死了的单位
                enemy_hero_attack_target[7] = 1 
        
        # 判断防御塔攻击目标
        main_first_tow_attack_target = [0,0,0]  # 攻击小兵 英雄 阵亡单位
        enemy_first_tow_attack_target = [0,0,0]  # 攻击小兵 英雄 阵亡单位
        main_second_tow_attack_target = [0,0,0]  # 攻击小兵 英雄 阵亡单位
        enemy_second_tow_attack_target = [0,0,0]  # 攻击小兵 英雄 阵亡单位
        if main_first_tow['attack_target'] != 0:
            try:
                attack_target_config_id = runtime_config_id_dict[main_first_tow['attack_target']]
                if attack_target_config_id in [6800, 6803, 6801, 6804, 6802, 6805]:
                    main_first_tow_attack_target[0] = 1
                elif attack_target_config_id in [133, 199, 508]:
                   main_first_tow_attack_target[1] = 1
            except:
                main_first_tow_attack_target[2] = 1
        if main_second_tow['attack_target'] != 0:
            try:
                attack_target_config_id = runtime_config_id_dict[main_second_tow['attack_target']]
                if attack_target_config_id in [6800, 6803, 6801, 6804, 6802, 6805]:
                    main_second_tow_attack_target[0] = 1
                elif attack_target_config_id in [133, 199, 508]:
                   main_second_tow_attack_target[1] = 1
            except:
                main_second_tow_attack_target[2] = 1
        if enemy_first_tow['attack_target'] != 0:
            try:
                attack_target_config_id = runtime_config_id_dict[enemy_first_tow['attack_target']]
                if attack_target_config_id in [6800, 6803, 6801, 6804, 6802, 6805]:
                    enemy_first_tow_attack_target[0] = 1
                elif attack_target_config_id in [133, 199, 508]:
                   enemy_first_tow_attack_target[1] = 1
            except:
                enemy_first_tow_attack_target[2] = 1
        if enemy_second_tow['attack_target'] != 0:
            try:
                attack_target_config_id = runtime_config_id_dict[enemy_second_tow['attack_target']]
                if attack_target_config_id in [6800, 6803, 6801, 6804, 6802, 6805]:
                    enemy_second_tow_attack_target[0] = 1
                elif attack_target_config_id in [133, 199, 508]:
                   enemy_second_tow_attack_target[1] = 1
            except:
                enemy_second_tow_attack_target[2] = 1

        # 整合
        feature_vec = np.concatenate([
            feature_vec, 
            np.array(cake_feature), 
            np.array(bullet_feature), 
            np.array(buff_feature), 
            np.array(main_hero_ation_list + enemy_hero_ation_list),
            np.array(main_hero_attack_target + enemy_hero_attack_target + main_first_tow_attack_target + main_second_tow_attack_target + enemy_first_tow_attack_target + enemy_second_tow_attack_target)
        ])

        return ObsData(
            feature=feature_vec, legal_action=legal_action, lstm_cell=self.lstm_cell, lstm_hidden=self.lstm_hidden
        )

    @learn_wrapper
    def learn(self, list_sample_data):
        list_npdata = [sample_data.npdata for sample_data in list_sample_data]
        _input_datas = np.stack(list_npdata, axis=0)
        _input_datas = torch.from_numpy(_input_datas).to(self.device)
        results = {}

        data_list = list(_input_datas.split(self.cut_points, dim=1))
        for i, data in enumerate(data_list):
            data = data.reshape(-1)
            data_list[i] = data.float()

        seri_vec = data_list[0].reshape(-1, self.data_split_shape[0])
        feature, legal_action = seri_vec.split(
            [
                np.prod(self.seri_vec_split_shape[0]),
                np.prod(self.seri_vec_split_shape[1]),
            ],
            dim=1,
        )
        init_lstm_cell = data_list[-2]
        init_lstm_hidden = data_list[-1]

        feature_vec = feature.reshape(-1, self.seri_vec_split_shape[0][0])
        lstm_hidden_state = init_lstm_hidden.reshape(-1, self.lstm_real_size)
        lstm_cell_state = init_lstm_cell.reshape(-1, self.lstm_real_size)

        format_inputs = [feature_vec, lstm_hidden_state, lstm_cell_state]

        self.model.set_train_mode()
        self.optimizer.zero_grad()

        rst_list = self.model(format_inputs, inference=False)
        total_loss, info_list = self.model.compute_loss(data_list, rst_list)
        results["total_loss"] = total_loss.item()

        total_loss.backward()

        # grad clip
        if Config.USE_GRAD_CLIP:
            torch.nn.utils.clip_grad_norm_(self.parameters, Config.GRAD_CLIP_RANGE)

        self.optimizer.step()
        self.train_step += 1

        _info_list = []
        for info in info_list:
            if isinstance(info, list):
                _info = [i.item() for i in info]
            else:
                _info = info.item()
            _info_list.append(_info)
        if self.monitor:
            _, (value_loss, policy_loss, entropy_loss) = _info_list
            results["value_loss"] = round(value_loss, 2)
            results["policy_loss"] = round(policy_loss, 2)
            results["entropy_loss"] = round(entropy_loss, 2)
            self.monitor.put_data({os.getpid(): results})

    @save_model_wrapper
    def save_model(self, path=None, id="1"):
        # To save the model, it can consist of multiple files, and it is important to ensure that
        #  each filename includes the "model.ckpt-id" field.
        # 保存模型, 可以是多个文件, 需要确保每个文件名里包括了model.ckpt-id字段
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        torch.save(self.model.state_dict(), model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    @load_model_wrapper
    def load_model(self, path=None, id="1"):
        # When loading the model, you can load multiple files, and it is important to ensure that
        # each filename matches the one used during the save_model process.
        # 加载模型, 可以加载多个文件, 注意每个文件名需要和save_model时保持一致
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        if self.cur_model_name == model_file_path:
            self.logger.info(f"current model is {model_file_path}, so skip load model")
        else:
            self.model.load_state_dict(
                torch.load(
                    model_file_path,
                    map_location=self.device,
                )
            )
            self.cur_model_name = model_file_path
            self.logger.info(f"load model {model_file_path} successfully")

    def reset(self, hero_camp, player_id):
        self.hero_camp = hero_camp
        self.player_id = player_id
        self.lstm_hidden = np.zeros([self.lstm_real_size])
        self.lstm_cell = np.zeros([self.lstm_real_size])
        self.reward_manager = GameRewardManager(player_id)

    def update_status(self, obs_data, act_data):
        self.obs_data = obs_data
        self.act_data = act_data
        self.lstm_cell = act_data.lstm_cell
        self.lstm_hidden = act_data.lstm_hidden

    # get final executable actions
    def _sample_masked_action(self, logits, legal_action):
        """
        Sample actions from predicted logits and legal actions
        return: probability, stochastic and deterministic actions with additional []
        """
        """
        从预测的logits和合法动作中采样动作
        返回：以列表形式概率、随机和确定性动作
        """

        prob_list = []
        action_list = []
        d_action_list = []
        label_split_size = [sum(self.label_size_list[: index + 1]) for index in range(len(self.label_size_list))]
        legal_actions = np.split(legal_action, label_split_size[:-1])
        logits_split = np.split(logits, label_split_size[:-1])
        for index in range(0, len(self.label_size_list) - 1):
            probs = self._legal_soft_max(logits_split[index], legal_actions[index])
            prob_list += list(probs)
            sample_action = self._legal_sample(probs, use_max=False)
            action_list.append(sample_action)
            d_action = self._legal_sample(probs, use_max=True)
            d_action_list.append(d_action)

        # deals with the last prediction, target
        # 处理最后的预测，目标
        index = len(self.label_size_list) - 1
        target_legal_action_o = np.reshape(
            legal_actions[index],  # [12, 8]
            [
                self.legal_action_size[0],
                self.legal_action_size[-1] // self.legal_action_size[0],
            ],
        )
        one_hot_actions = np.eye(self.label_size_list[0])[action_list[0]]  # [12]
        one_hot_actions = np.reshape(one_hot_actions, [self.label_size_list[0], 1])  # [12, 1]
        target_legal_action = np.sum(target_legal_action_o * one_hot_actions, axis=0)

        legal_actions[index] = target_legal_action  # [12]
        probs = self._legal_soft_max(logits_split[-1], target_legal_action)
        prob_list += list(probs)
        sample_action = self._legal_sample(probs, use_max=False)
        action_list.append(sample_action)

        # target_legal_action = tf.gather(target_legal_action, action_idx, axis=1)
        one_hot_actions = np.eye(self.label_size_list[0])[d_action_list[0]]
        one_hot_actions = np.reshape(one_hot_actions, [self.label_size_list[0], 1])
        target_legal_action_d = np.sum(target_legal_action_o * one_hot_actions, axis=0)

        # legal_actions[index] = target_legal_action
        probs = self._legal_soft_max(logits_split[-1], target_legal_action_d)

        # prob_list.append(probs)
        d_action = self._legal_sample(probs, use_max=True)
        d_action_list.append(d_action)

        return [prob_list], action_list, d_action_list

    def _legal_soft_max(self, input_hidden, legal_action):
        _lsm_const_w, _lsm_const_e = 1e20, 1e-5
        _lsm_const_e = 0.00001

        tmp = input_hidden - _lsm_const_w * (1.0 - legal_action)
        tmp_max = np.max(tmp, keepdims=True)
        # Not necessary max clip 1
        tmp = np.clip(tmp - tmp_max, -_lsm_const_w, 1)
        # tmp = tf.exp(tmp - tmp_max)* legal_action + _lsm_const_e
        tmp = (np.exp(tmp) + _lsm_const_e) * legal_action
        # tmp_sum = tf.reduce_sum(tmp, axis=1, keepdims=True)
        probs = tmp / np.sum(tmp, keepdims=True)
        return probs

    def _legal_sample(self, probs, legal_action=None, use_max=False):
        # Sample with probability, input probs should be 1D array
        # 根据概率采样，输入的probs应该是一维数组
        if use_max:
            return np.argmax(probs)

        return np.argmax(np.random.multinomial(1, probs, size=1))