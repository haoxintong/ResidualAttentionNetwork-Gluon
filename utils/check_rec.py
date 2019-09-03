# MIT License
# Copyright (c) 2019 haoxintong
""""""
import os
import numpy as np
from mxnet import base
from mxnet.gluon.data import dataset
from mxnet.gluon.data.vision import ImageRecordDataset


class CIFAR10(dataset._DownloadedDataset):
    """CIFAR10 image classification dataset from https://www.cs.toronto.edu/~kriz/cifar.html

    Each sample is an image (in 3D NDArray) with shape (32, 32, 3).

    Parameters
    ----------
    root : str, default $MXNET_HOME/datasets/cifar10
        Path to temp folder for storing data.
    train : bool, default True
        Whether to load the training or testing set.
    transform : function, default None
        A user defined callback that transforms each sample. For example::

            transform=lambda data, label: (data.astype(np.float32)/255, label)

    """

    def __init__(self, root=os.path.join(base.data_dir(), 'datasets', 'cifar10'),
                 train=True, transform=None):
        self._train = train
        self._archive_file = ('cifar-10-binary.tar.gz', 'fab780a1e191a7eda0f345501ccd62d20f7ed891')
        self._train_data = [('data_batch_1.bin', 'aadd24acce27caa71bf4b10992e9e7b2d74c2540'),
                            ('data_batch_2.bin', 'c0ba65cce70568cd57b4e03e9ac8d2a5367c1795'),
                            ('data_batch_3.bin', '1dd00a74ab1d17a6e7d73e185b69dbf31242f295'),
                            ('data_batch_4.bin', 'aab85764eb3584312d3c7f65fd2fd016e36a258e'),
                            ('data_batch_5.bin', '26e2849e66a845b7f1e4614ae70f4889ae604628')]
        self._test_data = [('test_batch.bin', '67eb016db431130d61cd03c7ad570b013799c88c')]
        self._namespace = 'cifar10'
        super(CIFAR10, self).__init__(root, transform)

    def _read_batch(self, filename):
        with open(filename, 'rb') as fin:
            data = np.frombuffer(fin.read(), dtype=np.uint8).reshape(-1, 3072 + 1)

        return data[:, 1:].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), \
               data[:, 0].astype(np.int32)

    def _get_data(self):

        if self._train:
            data_files = self._train_data
        else:
            data_files = self._test_data
        data, label = zip(*(self._read_batch(os.path.join(self._root, name))
                            for name, _ in data_files))
        data = np.concatenate(data)
        label = np.concatenate(label)

        self._data = data
        self._label = label


if __name__ == '__main__':

    cifar_ori = CIFAR10(train=False)
    cifar_rec = ImageRecordDataset("./cifar10_test.rec")

    for a, b in zip(cifar_ori, cifar_rec):
        im1 = a[0]
        im2 = b[0].asnumpy()[..., ::-1]
        diff = im1 - im2
        assert np.sum(diff) == 0
        assert a[1] == b[1]
