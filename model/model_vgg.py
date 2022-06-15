#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：CDINet-master
@File ：model_vgg16.py
@Author ：chen.zhang
@Date ：2021/2/1 9:50
"""

import mindspore.nn as nn
from backbone.VGG import VGG16
from modules.EM import RDE, DSE
from modules.DDR import DDR


class CDINet(nn.Cell):

    def __init__(self):
        super(CDINet, self).__init__()

        self.vgg_r = VGG16()
        self.vgg_d = VGG16()

        self.channels = [64, 128, 256, 512, 512]
        self.rde_layer1 = RDE(channel=self.channels[0])
        self.rde_layer2 = RDE(channel=self.channels[1])
        self.dse_layer3 = DSE(channel=self.channels[2], ratio=4)
        self.dse_layer4 = DSE(channel=self.channels[3], ratio=4)
        self.dse_layer5 = DSE(channel=self.channels[4], ratio=4)

        self.conv_mid = nn.Conv2d(self.channels[4], self.channels[4], kernel_size=3)
        self.relu_mid = nn.ReLU()

        self.ddr = DDR(self.channels)

    def construct(self, image, depth):
        """
        Args:
            image: The input of RGB images, three channels.
            depth: The input of Depth images, single channels.

        Returns: The final saliency maps.

        """
        decoder_list = []
        # image = data[:, 0, :].squeeze()
        # print(image.shape)
        # depth = data[:, 1, :].squeeze()
        image = image.squeeze(axis=1)
        depth = depth.squeeze(axis=1)

        conv1_vgg_r = self.vgg_r.conv1(image)
        conv1_vgg_d = self.vgg_d.conv1(depth)

        conv2_vgg_d_in = self.rde_layer1(conv1_vgg_r, conv1_vgg_d)
        decoder_list.append(conv2_vgg_d_in)
        conv2_vgg_r = self.vgg_r.conv2(conv1_vgg_r)
        conv2_vgg_d = self.vgg_d.conv2(conv2_vgg_d_in)

        conv3_vgg_d_in = self.rde_layer2(conv2_vgg_r, conv2_vgg_d)
        decoder_list.append(conv3_vgg_d_in)
        conv3_vgg_r = self.vgg_r.conv3(conv2_vgg_r)
        conv3_vgg_d = self.vgg_d.conv3(conv3_vgg_d_in)

        conv4_vgg_r_in, conv4_vgg_d_in = self.dse_layer3(conv3_vgg_r, conv3_vgg_d)
        decoder_list.append(conv4_vgg_r_in)
        conv4_vgg_r = self.vgg_r.conv4(conv4_vgg_r_in)
        conv4_vgg_d = self.vgg_d.conv4(conv4_vgg_d_in)

        conv5_vgg_r_in, conv5_vgg_d_in = self.dse_layer4(conv4_vgg_r, conv4_vgg_d)
        decoder_list.append(conv5_vgg_r_in)
        conv5_vgg_r = self.vgg_r.conv5(conv5_vgg_r_in)
        conv5_vgg_d = self.vgg_d.conv5(conv5_vgg_d_in)

        conv5_vgg_r_out, conv5_vgg_d_out = self.dse_layer5(conv5_vgg_r, conv5_vgg_d)
        mid_feature = self.relu_mid(self.conv_mid(conv5_vgg_r_out))
        decoder_list.append(mid_feature)

        smap = self.ddr(decoder_list)
        return smap


if __name__ == '__main__':
    import numpy as np
    import mindspore
    x = np.random.randn(2, 3, 256, 256)
    x = mindspore.Tensor(x, mindspore.float32)
    y = np.random.randn(2, 3, 256, 256)
    y = mindspore.Tensor(y, mindspore.float32)
    model = CDINet()
    model(x, y)
