#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：CDINet-master
@File ：DDR.py
@Author ：chen.zhang
@Date ：2021/2/1 9:55
"""

import numpy as np
from typing import List

import mindspore
from mindspore import Tensor
import mindspore.nn as nn
import mindspore.ops as ops


class BaseConv2d(nn.Cell):
    def __init__(self, in_planes, out_planes, kernel_size=3):
        super(BaseConv2d, self).__init__()
        self.basicconv = nn.SequentialCell(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size),
            nn.ReLU()
        )

    def construct(self, x):
        return self.basicconv(x)


class DDR(nn.Cell):
    def __init__(self, channels: List[int]) -> None:
        """
        Args:
            channels: It should a list which denotes the same channels
                      of encoder side outputs(skip connection features).
        """
        super(DDR, self).__init__()

        # decoder layer 5
        self.conv5 = nn.SequentialCell(
            BaseConv2d(channels[4], channels[4]),
            BaseConv2d(channels[4], channels[4]),
            BaseConv2d(channels[4], channels[3])

        )
        self.resize_5 = ops.ResizeBilinear((32, 32))

        # decoder layer 4
        self.conv4 = nn.SequentialCell(
            BaseConv2d(channels[3] * 2, channels[3]),
            BaseConv2d(channels[3], channels[3]),
            BaseConv2d(channels[3], channels[2])
        )
        self.resize_4 = ops.ResizeBilinear((64, 64))

        # decoder layer 3
        self.conv3 = nn.SequentialCell(
            BaseConv2d(channels[2] * 2, channels[2]),
            BaseConv2d(channels[2], channels[2]),
            BaseConv2d(channels[2], channels[1])
        )
        self.resize_3 = ops.ResizeBilinear((128, 128))

        # decoder layer 2
        self.conv2 = nn.SequentialCell(
            BaseConv2d(channels[1] * 2, channels[1]),
            BaseConv2d(channels[1], channels[0])
        )
        self.resize_2 = ops.ResizeBilinear((256, 256))

        # decoder layer 1
        self.conv1 = nn.SequentialCell(
            BaseConv2d(channels[0] * 2, channels[0]),
            BaseConv2d(channels[0], 3)
        )

        self.c1 = nn.SequentialCell(
            BaseConv2d(channels[4], channels[3], kernel_size=1),
            nn.Conv2d(channels[3], channels[3], kernel_size=3)
        )

        self.c2 = nn.SequentialCell(
            BaseConv2d(channels[4] + channels[3], channels[2], kernel_size=1),
            nn.Conv2d(channels[2], channels[2], kernel_size=3)
        )

        self.c3 = nn.SequentialCell(
            BaseConv2d(channels[4] + channels[3] + channels[2], channels[1], kernel_size=1),
            nn.Conv2d(channels[1], channels[1], kernel_size=3)
        )

        self.c4 = nn.SequentialCell(
            BaseConv2d(channels[4] + channels[3] + channels[2] + channels[1], channels[0],
                       kernel_size=1),
            nn.Conv2d(channels[0], channels[0], kernel_size=3)
        )
        self.conv_map = nn.Conv2d(3, 1, kernel_size=3)
        self.concat_all = ops.Concat(axis=1)

    def construct(self, decoder_list: List[Tensor]) -> Tensor:

        # assert len(decoder_list) == 5
        # decoder layer 5
        decoder_map5 = self.conv5(decoder_list[4])
        decoder_map5 = self.resize_5(decoder_map5)

        # decoder layer 4
        block4 = self.c1(self.resize_5(decoder_list[4]))
        short4 = block4 * decoder_list[3] + decoder_list[3]
        decoder_map4_input = self.concat_all([decoder_map5, short4])
        decoder_map4 = self.conv4(decoder_map4_input)
        decoder_map4 = self.resize_4(decoder_map4)

        # decoder layer 3
        block3 = self.c2(self.concat_all([self.resize_4(decoder_list[4]), self.resize_4(decoder_list[3])]))
        short3 = block3 * decoder_list[2] + decoder_list[2]
        decoder_map3_input = self.concat_all([decoder_map4, short3])
        decoder_map3 = self.conv3(decoder_map3_input)
        decoder_map3 = self.resize_3(decoder_map3)

        # decoder layer 2
        block2 = self.c3(self.concat_all([self.resize_3(decoder_list[4]),
                                          self.resize_3(decoder_list[3]),
                                          self.resize_3(decoder_list[2])]))
        short2 = block2 * decoder_list[1] + decoder_list[1]
        decoder_map2_input = self.concat_all([decoder_map3, short2])
        decoder_map2 = self.conv2(decoder_map2_input)
        decoder_map2 = self.resize_2(decoder_map2)

        # decoder layer 1
        block1 = self.c4(self.concat_all([self.resize_2(decoder_list[4]),
                                          self.resize_2(decoder_list[3]),
                                          self.resize_2(decoder_list[2]),
                                          self.resize_2(decoder_list[1])]))
        short1 = block1 * decoder_list[0] + decoder_list[0]
        decoder_map1_input = self.concat_all([decoder_map2, short1])
        decoder_map1 = self.conv1(decoder_map1_input)
        # decoder_map1 = self.resize_1(decoder_map1)c
        smap = self.conv_map(decoder_map1)

        return smap


if __name__ == '__main__':
    from mindspore import context
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')

    decoder_list = []
    x1 = mindspore.Tensor(np.ones([1, 64, 256, 256]).astype(np.float32), mindspore.float32)
    decoder_list.append(x1)
    x2 = mindspore.Tensor(np.ones([1, 128, 128, 128]).astype(np.float32), mindspore.float32)
    decoder_list.append(x2)
    x3 = mindspore.Tensor(np.ones([1, 256, 64, 64]).astype(np.float32), mindspore.float32)
    decoder_list.append(x3)
    x4 = mindspore.Tensor(np.ones([1, 512, 32, 32]).astype(np.float32), mindspore.float32)
    decoder_list.append(x4)
    x5 = mindspore.Tensor(np.ones([1, 512, 16, 16]).astype(np.float32), mindspore.float32)
    decoder_list.append(x5)

    channels = [64, 128, 256, 512, 512]
    ddr = DDR(channels)
    out = ddr(decoder_list)
    print(out.shape)
