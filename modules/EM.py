#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：CDINet-master
@File ：EM.py
@Author ：chen.zhang
@Date ：2021/8/1 9:52
"""
import numpy as np
from typing import List

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor


class ChannelAttention(nn.Cell):
    """
    The implementation of channel attention mechanism.
    """

    def __init__(self, channel: int, ratio: int = 4) -> None:
        """
        Args:
            channel: Number of channels for the input features.
            ratio: The node compression ratio in the full connection layer.
        """
        super(ChannelAttention, self).__init__()
        self.avg_pool = ops.ReduceMean(keep_dims=True)
        self.reshape = ops.Reshape()
        self.fc = nn.SequentialCell(
            nn.Dense(channel, channel // ratio),
            nn.ReLU(),
            nn.Dense(channel // ratio, channel)
        )
        self.sigmoid = nn.Sigmoid()

    def construct(self, x: Tensor) -> Tensor:
        """
        Returns the feature after passing the channel attention.
        """
        b, c, _, _ = x.shape
        y = self.avg_pool(x, axis=(2, 3))
        y = self.reshape(y, (b, c))
        y = self.fc(y)
        y = self.reshape(y, (b, c, 1, 1))
        y = self.sigmoid(y)

        out = x*y
        return out


class SpatialAttention(nn.Cell):
    """
    spatial attention, return weight map(default)
    """

    def __init__(self, kernel_size: int = 7) -> None:
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size)
        self.sigmoid = nn.Sigmoid()
        self.max = ops.ReduceMean(keep_dims=True)

    def construct(self, x: Tensor) -> Tensor:
        """
        Args:
            x: The input feature.

        Returns: A weight map of spatial attention, the size is H×W.

        """
        max_out = self.max(x, axis=1)
        x = self.conv1(max_out)
        weight_map = self.sigmoid(x)
        return weight_map


class RDE(nn.Cell):
    """
    The implementation of RGB-induced details enhancement module.
    """

    def __init__(self, channel: int) -> None:
        super(RDE, self).__init__()
        self.cat = ops.Concat(axis=1)
        self.conv_pool = nn.SequentialCell(
            nn.Conv2d(channel * 2, channel, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size=3)
        )
        self.max = ops.ReduceMean(keep_dims=True)
        self.conv1 = nn.Conv2d(1, 1, kernel_size=7)
        self.conv2 = nn.Conv2d(1, 1, kernel_size=7)
        self.sigmoid = ops.Sigmoid()

    def construct(self, input_rgb: Tensor, input_depth: Tensor) -> Tensor:
        rgbd = self.cat([input_rgb, input_depth])
        feature_pool = self.conv_pool(rgbd)
        x = self.max(input_depth, axis=1)
        x = self.conv2(self.conv1(x))
        mask = self.sigmoid(x)

        depth_enhance = feature_pool * mask + input_depth

        return depth_enhance


class DSE(nn.Cell):
    """
    The implementation of depth-induced semantic enhancement module.
    """

    def __init__(self, channel: int, ratio: int = 4) -> None:
        super(DSE, self).__init__()
        self.sa1 = SpatialAttention(kernel_size=3)
        self.sa2 = SpatialAttention(kernel_size=3)

        self.ca1 = ChannelAttention(channel, ratio)
        self.ca2 = ChannelAttention(channel, ratio)

    def construct(self, input_rgb: Tensor, input_depth: Tensor) -> (Tensor, Tensor):
        map_depth = self.sa1(input_depth)
        input_rgb_sa = input_rgb*map_depth + input_rgb
        input_rgb_sa_ca = self.ca1(input_rgb_sa)

        map_depth2 = self.sa2(input_depth)
        input_depth_sa = input_depth*map_depth2 + input_depth
        input_depth_sa_ca = self.ca2(input_depth_sa)

        return input_rgb_sa_ca + input_depth_sa_ca, input_depth_sa_ca


if __name__ == '__main__':
    x = np.random.randn(2, 4, 256, 256)
    x = mindspore.Tensor(x, mindspore.float32)
    y = np.random.randn(2, 4, 256, 256)
    y = mindspore.Tensor(y, mindspore.float32)
    net = DSE(4)

