#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2024 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from kaiwu_agent.utils.common_func import create_cls, Frame, attached
from diy.config import Config
import numpy as np
import collections
import random
import itertools
import os
import json

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

NONE_ACTION = [0, 15, 15, 15, 15, 0]


# Loop through camps, shuffling camps before each major loop
# 循环返回camps, 每次大循环前对camps进行shuffle
def _lineup_iterator_shuffle_cycle(camps):
    while True:
        random.shuffle(camps)
        for camp in camps:
            yield camp


# Specify single-side multi-agent lineups, looping through all pairwise combinations
# 指定单边多智能体阵容，两两组合循环
def lineup_iterator_roundrobin_camp_heroes(camp_heroes=None):
    if not camp_heroes:
        raise Exception(f"camp_heroes is empty")

    try:
        valid_ids = [133, 199, 508]
        for camp in camp_heroes:
            hero_id = camp[0]["hero_id"]
            if hero_id not in valid_ids:
                raise Exception(f"hero_id {hero_id} not valid, valid is {valid_ids}")
    except Exception as e:
        raise Exception(f"check hero valid, exception is {str(e)}")

    camps = []
    for lineups in itertools.product(camp_heroes, camp_heroes):
        camp = []
        for lineup in lineups:
            camp.append(lineup)
        camps.append(camp)
    return _lineup_iterator_shuffle_cycle(camps)


@attached
def sample_process(collector):
    return collector.sample_process()


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

    def reset(self, num_agents):
        self.num_agents = num_agents
        self.rl_data_map = [collections.OrderedDict() for _ in range(self.num_agents)]
        self.m_replay_buffer = [[] for _ in range(self.num_agents)]

    def sample_process(self):
        return


# SampleData <----> NumpyData
@attached
def SampleData2NumpyData(g_data):
    return g_data.npdata


@attached
def NumpyData2SampleData(s_data):
    return SampleData(npdata=s_data)
