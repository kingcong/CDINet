import argparse

parser = argparse.ArgumentParser()
# train set
parser.add_argument('--epoch', type=int, default=120,
                    help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--batchsize', type=int, default=4,
                    help='training batch size')
parser.add_argument('--trainsize', type=int, default=256,
                    help='training image size')
parser.add_argument('--clip', type=float, default=0.5,
                    help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.2,
                    help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=40,
                    help='every n epochs decay learning rate')
parser.add_argument('--load', type=str, default=None,
                    help='train from checkpoints')
parser.add_argument("--device_id", type=str, default='0', help="Device id")
parser.add_argument('--device_target', type=str, default="Ascend",
                    choices=['Ascend', 'GPU', 'CPU'], help='device target')
parser.add_argument('--rgb_root', type=str, default='/cache/data_path/RGBD_SOD_Dataset/CDINet_train_data/RGB/',
                    help='the training rgb images root')
parser.add_argument('--depth_root', type=str, default='/cache/data_path/RGBD_SOD_Dataset/CDINet_train_data/depth/',
                    help='the training depth images root')
parser.add_argument('--gt_root', type=str, default='/cache/data_path/RGBD_SOD_Dataset/CDINet_train_data/GT/',
                    help='the training gt images root')
parser.add_argument('--save_path', type=str, default='/cache/data_path/Checkpoints/',
                    help='the path to save models and logs')
# test set
parser.add_argument('--testsize', type=int, default=256,
                    help='testing image size')
parser.add_argument('--test_path', type=str, default='/cache/data_path/CDINet_test_data/RGBD_for_test/',
                    help='test dataset path')
