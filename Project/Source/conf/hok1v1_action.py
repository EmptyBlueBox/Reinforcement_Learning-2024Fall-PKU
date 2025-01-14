#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2024 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import numpy as np
from kaiwudrl.interface.array_spec import ArraySpec
from kaiwudrl.common.algorithms.distribution import CategoricalDist
from kaiwudrl.interface.action import Action, ActionSpec


class SgameAction(Action):
    def __init__(self, a):
        self.a = a

    def get_action(self):
        return {"a": self.a}

    @staticmethod
    def action_space():
        action_space = 50
        return {"a": ActionSpec(ArraySpec((action_space,), np.int32), pdclass=CategoricalDist)}

    def __str__(self):
        return str(self.a)
