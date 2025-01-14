#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2024 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import os
import time
import random
import numpy as np
from ppo.feature.definition import (
    sample_process,
    build_frame,
    lineup_iterator_roundrobin_camp_heroes,
    FrameCollector,
    NONE_ACTION,
)
from kaiwu_agent.utils.common_func import attached
from ppo.config import GameConfig
from tools.model_pool_utils import get_valid_model_pool

extra_set_all = set()


@attached
def workflow(envs, agents, logger=None, monitor=None):
    # Whether the agent is training, corresponding to do_predicts
    # 智能体是否进行训练
    do_learns = [True, True]
    last_save_model_time = time.time()

    while True:
        for g_data in run_episodes(envs, agents, logger, monitor):
            for index, (d_learn, agent) in enumerate(zip(do_learns, agents)):
                if d_learn and len(g_data[index]) > 0:
                    # The learner trains in a while true loop, here learn actually sends samples
                    # learner 采用 while true 训练，此处 learn 实际为发送样本
                    agent.learn(g_data[index])
            g_data.clear()

            now = time.time()
            if now - last_save_model_time > GameConfig.MODEL_SAVE_INTERVAL:
                agents[0].save_model()
                last_save_model_time = now


def run_episodes(envs, agents, logger, monitor):
    global extra_set_all
    # hok1v1 environment
    # hok1v1环境
    env = envs[0]
    # Number of agents, in hok1v1 the value is 2
    # 智能体数量，在hok1v1中值为2
    agent_num = len(agents)
    # Episode counter
    # 对局数量计数器
    episode_cnt = 0
    # ID of Agent to training
    # 每一局要训练的智能体的id
    train_agent_id = 0
    # Lineup iterator
    # 阵容生成器
    lineup_iter = lineup_iterator_roundrobin_camp_heroes(camp_heroes=GameConfig.CAMP_HEROES)
    # Frame Collector
    # 帧收集器
    frame_collector = FrameCollector(agent_num)     # 详情见ppo/feature/definition.py
    # Make eval matches as evenly distributed as possible
    # 引入随机因子，让eval对局尽可能平均分布
    random_eval_start = random.randint(0, GameConfig.EVAL_FREQ)

    # Single environment process (30 frame/s)
    # 单局流程 (30 frame/s)
    while True:
        high_difficulty_samples = []    # 高价值样本多学习？

        # Settings before starting a new environment
        # 以下是启动一个新对局前的设置

        # Set the id of the agent to be trained. id=0 means the blue side, id=1 means the red side.
        # 设置要训练的智能体的id，id=0表示蓝方，id=1表示红方，每一局都切换一次阵营。默认对手智能体是selfplay即自己
        train_agent_id = 1 - train_agent_id
        opponent_agent = "selfplay"

        # Evaluate at a certain frequency during training to reflect the improvement of the agent during training
        # 智能体支持边训练边评估，训练中按一定的频率进行评估，反映智能体在训练中的水平
        is_eval = (episode_cnt + random_eval_start) % GameConfig.EVAL_FREQ == 0
        if is_eval:
            # The model used by the opponent: "common_ai" - rule-based agent, model_id - opponent model ID, see kaiwu.json for details
            # 设置评估时的对手智能体类型，默认采用了common_ai，可选择: "common_ai" - 基于规则的智能体, model_id - 对手模型的ID, 模型ID内容可在kaiwu.json里查看和设置
            opponent_agent = "common_ai"
            # opponent_agent = "24259"
            # opponent_agent = np.random.choice(["common_ai", "29946", "30470"])

        # Generate a new set of agent configurations
        # 生成一组新的智能体配置
        heroes_config = next(lineup_iter)

        usr_conf = {
            "diy": {
                # The side reporting the environment metrics
                # 上报对局指标的阵营
                "monitor_side": train_agent_id,     # 设置要训练的智能体的id，id=0表示蓝方，id=1表示红方
                # The label for reporting environment metrics: selfplay - "selfplay", common_ai - "common_ai", opponent model - model_id
                # 上报对局指标的标签： 自对弈 - "selfplay", common_ai - "common_ai", 对手模型 - model_id
                "monitor_label": opponent_agent,    # 对手模型
                # Indicates the lineups used by both sides
                # 表示双方使用的阵容
                "lineups": heroes_config,           # 智能体对手配置
                # "lineups": [[{'hero_id': 199}], [{'hero_id': 199}]],
            }
        }

        if train_agent_id not in [0, 1]:
            raise Exception("monitor_side is not valid, valid monitor_side list is [0, 1], please check")

        # Start a new environment
        # 启动新对局，返回初始环境状态
        _, state_dicts = env.reset(usr_conf=usr_conf)
        if state_dicts is None:
            logger.info(f"episode {episode_cnt}, reset is None happened!")
            continue

        # Game variables
        # 对局变量
        episode_cnt += 1
        frame_no = 0
        step = 0
        # Record the cumulative rewards of the agent in the environment
        # 记录对局中智能体的累积回报，用于上报监控
        total_reward_dicts = [{}, {}]
        logger.info(f"Episode {episode_cnt} start, usr_conf is {usr_conf}")

        # Reset agent
        # 重置agent

        # The 'do_predicts' specifies which agents are to perform model predictions.
        # Since the default opponent model is 'selfplay', it is set to [True, True] by default.
        # do_predicts指定哪些智能体要进行模型预测，由于默认对手模型是selfplay，默认设置[True, True]
        do_predicts = [True, True]
        for i, agent in enumerate(agents):
            player_id = state_dicts[i]["player_id"]
            camp = state_dicts[i]["player_camp"]
            agent.reset(camp, player_id)

            # The agent to be trained should load the latest model
            # 要训练的智能体 应 加载最新的模型
            if i == train_agent_id:
                # train_agent_id uses the latest model
                # train_agent_id 使用最新模型
                agent.load_model(id="latest")
            else:
                if opponent_agent == "common_ai":
                    # common_ai does not need to load a model, no need to predict
                    # 如果对手是 common_ai 则不需要加载模型, 也不需要进行预测
                    do_predicts[i] = False
                elif opponent_agent == "selfplay":
                    # Training model, "latest" - latest model, "random" - random model from the model pool
                    # 加载训练过的模型，可以选择最新模型，也可以选择随机模型 "latest" - 最新模型, "random" - 模型池中随机模型
                    agent.load_model(id="latest")
                else:
                    # Opponent model, model_id is checked from kaiwu.json
                    # 选择kaiwu.json中设置的对手模型, model_id 即 opponent_agent，必须设置正确否则报错
                    eval_candidate_model = get_valid_model_pool(logger)
                    if int(opponent_agent) not in eval_candidate_model:
                        raise Exception(f"model_id {opponent_agent} not in {eval_candidate_model}")
                    else:
                        agent.load_model(id=opponent_agent)

            logger.info(f"agent_{i} reset playerid:{player_id} camp:{camp}")

        # Reward initialization
        # 回报初始化，作为当前环境状态state_dicts的一部分
        for i in range(agent_num):
            reward = agents[i].reward_manager.result(state_dicts[i]["frame_state"])
            state_dicts[i]["reward"] = reward
            for key, value in reward.items():
                if key in total_reward_dicts[i]:
                    total_reward_dicts[i][key] += value
                else:
                    total_reward_dicts[i][key] = value

        # Reset environment frame collector
        # 重置环境帧收集器
        frame_collector.reset(num_agents=agent_num)

        while True:
            # Initialize the default actions. If the agent does not make a decision, env.step uses the default action.
            # 初始化默认的actions，如果智能体不进行决策，则env.step使用默认action
            actions = [
                NONE_ACTION,
            ] * agent_num

            for index, (d_predict, agent) in enumerate(zip(do_predicts, agents)):
                if d_predict:
                    if not is_eval:
                        actions[index] = agent.train_predict(state_dicts[index])
                    else:
                        actions[index] = agent.eval_predict(state_dicts[index])

                    # Only when do_predict=True and is_eval=False, the agent's environment data is saved.
                    # 仅do_predict=True且is_eval=False时，智能体的对局数据保存。即评估对局数据不训练，不是最新模型产生的数据不训练
                    if not is_eval:
                        frame = build_frame(agent, state_dicts[index])
                        frame_collector.save_frame(frame, agent_id=index)
                    # frame = build_frame(agent, state_dicts[index])
                    # frame_collector.save_frame(frame, agent_id=index)

            """
            The format of action is like [[2, 10, 1, 14, 8, 0], [1, 3, 10, 10, 9, 0]]
            There are 2 agents, so the length of the array is 2, and the order of values in
            each element is: button, move (2), skill (2), target
            action格式形如[[2, 10, 1, 14, 8, 0], [1, 3, 10, 10, 9, 0]]
            2个agent, 故数组的长度为2, 每个元素里面的值的顺序是:button, move(2个), skill(2个), target
            """

            # Step forward
            # 推进环境到下一帧，得到新的状态
            # actions[0][1] = 16
            # actions[0][2] = 16
            # actions[1][1] = 16
            # actions[1][2] = 16
            frame_no, _, _, terminated, truncated, state_dicts = env.step(actions)

            # Disaster recovery
            # 容灾
            if state_dicts is None:
                logger.info(f"episode {episode_cnt}, step({step}) is None happened!")
                break

            # Reward generation
            # 计算回报，作为当前环境状态state_dicts的一部分
            for i in range(agent_num):
                reward = agents[i].reward_manager.result(state_dicts[i]["frame_state"])
                state_dicts[i]["reward"] = reward
                for key, value in reward.items():
                    if key in total_reward_dicts[i]:
                        total_reward_dicts[i][key] += value
                    else:
                        total_reward_dicts[i][key] = value

            step += 1

            # 调试用
            # if len(frame_collector) > 64:
            #     list_agents_samples = sample_process(frame_collector)
            #     yield list_agents_samples

            # Normal end or timeout exit
            # 正常结束或超时退出
            if terminated or truncated:
                if len(agents[0].extra_set) > 0:
                    last_len = len(extra_set_all)
                    extra_set_all = extra_set_all.union(agents[0].extra_set)
                    if len(extra_set_all) > last_len:
                        logger.info(f"extra_buff: {len(extra_set_all)} =>{extra_set_all}")
                logger.info(
                    f"episode_{episode_cnt} terminated in fno_{frame_no}, truncated:{truncated}, eval:{is_eval}, total_reward_dicts:{total_reward_dicts}"
                )
                # Reward for saving the last state of the environment
                # 保存环境最后状态的reward
                for index, (d_predict, agent) in enumerate(zip(do_predicts, agents)):
                    if d_predict and not is_eval:
                        frame_collector.save_last_frame(
                            agent_id=index,
                            reward=state_dicts[index]["reward"]["reward_sum"],
                        )

                monitor_data = {
                    "reward": round(total_reward_dicts[train_agent_id]["reward_sum"], 2),       # 总奖励
                    "diy_1": round(total_reward_dicts[train_agent_id]["forward"], 2),           # 向前奖励
                    "diy_2": round(total_reward_dicts[train_agent_id]["close_to_cake"], 2),     # 靠近血包奖励
                    "diy_3": round(total_reward_dicts[train_agent_id]["extra_mov_spd"], 2),     # 额外移速奖励
                    "diy_4": round(total_reward_dicts[train_agent_id]["hit_target"], 2),        # 攻击目标奖励
                    "diy_5": -round(total_reward_dicts[train_agent_id]["hit_by_organ"], 2),     # 被塔攻击奖励
                }

                if monitor and is_eval:
                    monitor.put_data({os.getpid(): monitor_data})

                # Sample process
                # 进行样本处理，准备训练
                if len(frame_collector) > 0 and not is_eval:
                    list_agents_samples = sample_process(frame_collector)
                    yield list_agents_samples
                break