#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：CDINet-master
@File ：dataloader.py
@Author ：chen.zhang
@Date ：2021/8/15 10:00
"""

import os
import random
import numpy as np
from PIL import Image

import mindspore
from mindspore.dataset.vision import Inter
import mindspore.dataset.vision.c_transforms as c_vision
import mindspore.dataset.vision.py_transforms as py_vision
from mindspore.dataset.transforms.py_transforms import Compose


class GetDatasetGenerator:

    def __init__(self, image_root, depth_root, gt_root, trainsize):
        """
        :param image_root: The path of RGB training images.
        :param depth_root: The path of depth training images.
        :param gt_root: The path of training ground truth.
        :param trainsize: The size of training images.
        """
        self.trainsize = trainsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')]
        self.depths = [depth_root + f for f in os.listdir(depth_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.depths = sorted(self.depths)
        self.gts = sorted(self.gts)
        self._filter_files()
        self.size = len(self.images)

        resize_op = c_vision.Resize(size=(self.trainsize, self.trainsize), interpolation=Inter.LINEAR)
        normal_op_rgb = c_vision.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        normal_op_depth = c_vision.Normalize([0.485, ], [0.229, ])
        toTensor_op = py_vision.ToTensor()
        transforms_list_rgb = [resize_op, normal_op_rgb, toTensor_op]
        self.compose_trans_rgb = Compose(transforms_list_rgb)
        transforms_list_depth = [resize_op, normal_op_depth, toTensor_op]
        self.compose_trans_depth = Compose(transforms_list_depth)
        transforms_list_gt = [resize_op, toTensor_op]
        self.compose_trans_gt = Compose(transforms_list_gt)

    def __getitem__(self, index):
        image = self._rgb_loader(self.images[index])
        depth = self._rgb_loader(self.depths[index])
        gt = self._binary_loader(self.gts[index])

        image, depth, gt = self._randomFlip(image, depth, gt)
        image, depth, gt = self._randomRotation(image, depth, gt)

        image = self.compose_trans_rgb(np.array(image))
        depth = self.compose_trans_depth(np.array(depth))
        gt = self.compose_trans_gt(np.array(gt))

        return image, depth, gt

    def __len__(self):
        return self.size

    def _filter_files(self):
        """ Check whether a set of images match in size. """
        assert len(self.images) == len(self.depths) == len(self.gts)
        images = []
        depths = []
        gts = []
        for img_path, depth_path, gt_path in zip(self.images, self.depths, self.gts):
            # Notes: On DUT dataset, the size of training depth images are [256, 256],
            # it is not matched with RGB images and GT [600, 400].
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                depths.append(depth_path)
                gts.append(gt_path)
            else:
                raise Exception("Image sizes do not match, please check.")
        self.images = images
        self.depths = depths
        self.gts = gts

    def _rgb_loader(self, path):
        with open(path, 'rb') as f:
            # Removing alpha channel.
            return Image.open(f).convert('RGB')

    def _binary_loader(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('L')

    def _randomFlip(self, img, depth, gt):
        flip_flag = random.randint(0, 2)
        if flip_flag == 1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
            gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
        elif flip_flag == 2:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            depth = depth.transpose(Image.FLIP_TOP_BOTTOM)
            gt = gt.transpose(Image.FLIP_TOP_BOTTOM)
        return img, depth, gt

    def _randomRotation(self, image, depth, gt):
        mode = Image.BICUBIC
        if random.random() > 0.8:
            random_angle = np.random.randint(-15, 15)
            image = image.rotate(random_angle, mode)
            depth = depth.rotate(random_angle, mode)
            gt = gt.rotate(random_angle, mode)
        return image, depth, gt


if __name__ == '__main__':

    image_root = '../../../数据集/CDINet_train_data/RGB/'
    depth_root = '../../../数据集/CDINet_train_data/depth/'
    gt_root = '../../../数据集/CDINet_train_data/GT/'

    dataset_generator = GetDatasetGenerator(image_root, depth_root, gt_root, 224)
    dataset = mindspore.dataset.GeneratorDataset(
              dataset_generator, ["rgb", "depth", "label"], shuffle=True, num_parallel_workers=4)
    dataset = dataset.batch(batch_size=4)
    print(dataset.get_batch_size())
    for i, (images, depths, gts) in enumerate(dataset, start=1):
        print(images.shape)
        break


