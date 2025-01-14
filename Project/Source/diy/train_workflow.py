#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2024 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from kaiwu_agent.utils.common_func import Frame, attached
import random

from diy.feature.definition import (
    sample_process,
    lineup_iterator_roundrobin_camp_heroes,
    FrameCollector,
    NONE_ACTION,
)
from diy.config import GameConfig
from tools.model_pool_utils import get_valid_model_pool


@attached
def workflow(envs, agents, logger=None, monitor=None):
    # hok1v1 environment
    # hok1v1环境
    env = envs[0]
    # Number of agents, in hok1v1 the value is 2
    # 智能体数量，在hok1v1中值为2
    agent_num = len(agents)
    # Lineup iterator
    # 阵容生成器
    lineup_iter = lineup_iterator_roundrobin_camp_heroes(camp_heroes=GameConfig.CAMP_HEROES)
    # Frame Collector
    # 帧收集器
    frame_collector = FrameCollector(agent_num)
    # Make eval matches as evenly distributed as possible
    # 引入随机因子，让eval对局尽可能平均分布
    random_eval_start = random.randint(0, GameConfig.EVAL_FREQ)

    # Please implement your DIY algorithm flow
    # 请实现你DIY的算法流程
    # ......

    # Single environment process (30 frame/s)
    # 单局流程 (30 frame/s)
    """
    while True:
        # Generate a new set of agent configurations
        # 生成一组新的智能体配置
        heroes_config = next(lineup_iter)
        pass
    """

    return
