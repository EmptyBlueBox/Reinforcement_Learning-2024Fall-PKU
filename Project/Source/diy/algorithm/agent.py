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

from kaiwu_agent.agent.base_agent import (
    predict_wrapper,
    exploit_wrapper,
    learn_wrapper,
    save_model_wrapper,
    load_model_wrapper,
    BaseAgent,
)
from diy.model.model import Model
from kaiwu_agent.utils.common_func import attached
from diy.feature.definition import *
from diy.config import Config
from diy.feature.reward_manager import GameRewardManager


@attached
class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        super().__init__(agent_type, device, logger, monitor)
        self.cur_model_name = ""
        self.device = device
        # Create Model and convert the model to a channel-last memory format to achieve better performance.
        # 创建模型, 将模型转换为通道后内存格式，以获得更好的性能。
        self.model = Model().to(self.device)
        self.model = self.model.to(memory_format=torch.channels_last)

        # env info
        self.hero_camp = 0
        self.player_id = 0
        self.game_id = None

        # tools
        self.reward_manager = None
        self.logger = logger
        self.monitor = monitor

    def _model_inference(self, list_obs_data):
        # Code to implement model inference
        # 实现模型推理的代码
        list_act_data = [ActData()]
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

    @learn_wrapper
    def learn(self, list_sample_data):
        # Code to implement model train
        # 实现模型训练的代码
        return

    def action_process(self, state_dict, act_data, is_stochastic):
        # Implement the conversion from ActData to action
        # 实现ActData到action的转换
        return act_data.action

    def observation_process(self, state_dict):
        # Implement the conversion from State to ObsData
        # 实现State到ObsData的转换
        return ObsData(feature=[], legal_action=[], lstm_cell=self.lstm_cell, lstm_hidden=self.lstm_hidden)

    def reset(self, hero_camp, player_id):
        self.hero_camp = hero_camp
        self.player_id = player_id
        self.reward_manager = GameRewardManager(player_id)

    def update_status(self, obs_data, act_data):
        self.obs_data = obs_data
        self.act_data = act_data

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
