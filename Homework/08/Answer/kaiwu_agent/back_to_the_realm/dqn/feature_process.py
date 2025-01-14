import numpy as np
from arena_proto.back_to_the_realm.custom_pb2 import RelativeDirection


def one_hot_encoding(grid_pos):
    """
    此函数将proto传输的网格位置特征进行one_hot_encoding处理, 返回一个长度为256的向量
        - 前128维是x轴的one-hot特征
        - 后128维是z轴的one-hot特征
    """
    one_hot_pos_x, one_hot_pos_z = np.zeros(128).tolist(), np.zeros(128).tolist()
    one_hot_pos_x[grid_pos.x], one_hot_pos_z[grid_pos.z] = 1, 1
    
    return one_hot_pos_x + one_hot_pos_z


def read_relative_position(rel_pos):
    """
    此函数将proto传输的相对位置特征进行拆包并处理, 返回一个长度为9的向量
        - 前8维是one-hot的方向特征
        - 最后一维是距离特征
    """
    direction = [0] * 8
    if rel_pos.direction != RelativeDirection.RELATIVE_DIRECTION_NONE:
        direction[rel_pos.direction - 1] = 1

    distance = rel_pos.grid_distance
    return direction + [distance]


def bump(a1, b1, a2, b2):
    """
    该函数用于判断是否撞墙
        - 第一帧不会bump
        - 第二帧开始, 如果移动距离小于500则视为撞墙
    """
    if a2 == -1 and b2 == -1:
        return False
    if a1 == -1 and b1 == -1:
        return False 

    dist = ((a1-a2)**2 + (b1-b2)**2) ** (0.5)

    return dist <= 500
