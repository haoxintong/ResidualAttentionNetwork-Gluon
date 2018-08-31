# MIT License
#
# Copyright (c) 2018 Haoxintong
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Reorganize cifar-10 data"""

import os
import cv2
import pickle
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Reorganize data.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-root', required=True, type=str,
                        help='root dir of data')
    parser.add_argument('--output-root', required=True, type=str,
                        help='output dir of the images')
    args = parser.parse_args()
    return args


files = ["data_batch_1", "data_batch_2", "data_batch_3",
         "data_batch_4", "data_batch_5", "test_batch"]


def load_pickle_data(file_name):
    with open(file_name, "rb") as f:
        data = pickle.load(f, encoding='bytes')
    return data


def mkdir_if_not_exist(path):
    if not os.path.exists(os.path.join(*path)):
        os.makedirs(os.path.join(*path))


def reorg_data(data_root, output_root):
    for file in files:
        im_root = os.path.join(output_root, "train" if not file.startswith("test") else "test")
        data = load_pickle_data(os.path.join(data_root, file))
        for i in range(len(data[b'labels'])):
            img = data[b'data'][i]
            label = data[b'labels'][i]
            filename = data[b'filenames'][i].decode()
            mkdir_if_not_exist([im_root, str(label)])
            img = np.reshape(img, [3, 32, 32])
            img = img.transpose((1, 2, 0))
            img = img[..., ::-1]
            cv2.imwrite(os.path.join(im_root, str(label), filename), img)


if __name__ == '__main__':
    pargs = parse_args()
    reorg_data(pargs.data_root, pargs.output_root)
