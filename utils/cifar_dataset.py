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
"""Dataset for load cifar-10"""

import os
import pickle
import numpy as np
from mxnet import nd
from mxnet.gluon.data import Dataset


class CifarDataset(Dataset):
    def __init__(self, root, is_train, flag=1, transform=None):
        self._file_names = ["data_batch_1", "data_batch_2", "data_batch_3",
                            "data_batch_4", "data_batch_5"] if is_train else ["test_batch"]
        self._root = os.path.expanduser(root)

        self._flag = flag
        self._transform = transform
        self._ext = '.png'
        self._list_images()

    def _list_images(self):
        self.items = []
        for file in self._file_names:
            data_path = os.path.join(self._root, file)
            with open(data_path, "rb") as f:
                data = pickle.load(f, encoding='bytes')
                for i in range(len(data[b'labels'])):
                    label = data[b'labels'][i]
                    img = data[b'data'][i]
                    img = np.reshape(img, [3, 32, 32])
                    img = img.transpose((1, 2, 0))
                    self.items.append((nd.array(img), label))

    def __getitem__(self, idx):
        img = self.items[idx][0]
        label = self.items[idx][1]
        if self._transform is not None:
            return self._transform(img, label)
        return img, label

    def __len__(self):
        return len(self.items)
