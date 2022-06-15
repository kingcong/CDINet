#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：CDINet-master
@File ：CDINet.py
@Author ：chen.zhang
@Date ：2021/8/11 10:00
"""

import os
import time
import random
import numpy as np
from datetime import datetime

import mindspore
import moxing as mox
from mindspore import nn, dataset, context
from mindspore.nn.dynamic_lr import piecewise_constant_lr

from setting.dataLoader import GetDatasetGenerator
from model.model_vgg import CDINet


# Random Seed
def seed_torch(seed=42):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def set_dynamic_lr(o_lr):
    milestone = [40, 80, 120]
    learning_rates = [o_lr*0.1, o_lr*0.01, o_lr*0.001]
    lr = piecewise_constant_lr(milestone, learning_rates)
    return lr


class ComputeLoss(nn.Cell):
    def __init__(self, network, loss_fn):
        super(ComputeLoss, self).__init__(auto_prefix=False)
        self.network = network
        self._loss_fn = loss_fn

    def construct(self, rgb, depth, label):
        label = label.squeeze(axis=1)
        out = self.network(rgb, depth)
        return self._loss_fn(out, label)


# Train Function
def train(opt):
    # data
    print('load data...')
    dataset_generator = GetDatasetGenerator(opt.rgb_root, opt.depth_root, opt.gt_root, opt.trainsize)
    dataset_train = dataset.GeneratorDataset(
                    dataset_generator, ["rgb", "depth", "label"], shuffle=True, num_parallel_workers=4)
    dataset_train = dataset_train.batch(opt.batchsize)
    iterations_epoch = dataset_train.get_dataset_size()
    train_iterator = dataset_train.create_dict_iterator()
    # model
    print('load model...')
    model = CDINet()
    # Restore training from checkpoints
    if opt.load is not None:
        mindspore.load_checkpoint(opt.load, net=model)
        print('load model from', opt.load)
    # Optimizer
    # optimizer = mindspore.nn.optim.Adam(model.trainable_params(), set_dynamic_lr(opt.lr))
    optimizer = mindspore.nn.optim.Adam(model.trainable_params(), opt.lr)
    # Loss Function
    bce_loss = mindspore.nn.BCEWithLogitsLoss()
    # train network
    net = ComputeLoss(model, bce_loss)
    T_net = nn.TrainOneStepCell(net, optimizer)
    model.set_train()

    epoch = opt.epoch
    print("==================Starting Training==================")
    for i in range(epoch):
        loss_all = 0
        epoch_num = i + 1
        time_begin_epoch = time.time()
        for iteration, data in enumerate(train_iterator, start=1):
            loss_step = T_net(data["rgb"], data["depth"], data["label"])
            loss_all += loss_step.asnumpy()
            if iteration % 50 == 0:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.
                      format(datetime.now(), epoch_num, epoch, iteration, iterations_epoch, loss_step.asnumpy()))
        time_end_epoch = time.time()
        print('Epoch [{:03d}/{:03d}]:Loss_AVG={:.4f}, Time:{:.2f}'.
              format(epoch_num, epoch, loss_all/iterations_epoch, time_end_epoch - time_begin_epoch))
        if epoch_num > 80 and epoch_num % 5 == 0:
            mindspore.save_checkpoint(model, opt.data_url+'Checkpoints/'+'CDINet_'+str(epoch_num))
    print("==================Ending Training==================")


if __name__ == '__main__':

    from setting.options import parser
    # OBS Configuration
    parser.add_argument('--data_url', required=True, default=None, help='Location of data.')
    parser.add_argument('--train_url', required=True, default=None, help='Location of training outputs.')
    args = parser.parse_args()
    mox.file.copy_parallel(src_url=args.data_url, dst_url='/cache/data_path')

    # Device Configuration
    os.environ['ASCEND_SLOG_PRINT_TO_STDOUT'] = args.device_id
    context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device_target)

    # Train
    seed_torch()
    train(args)

