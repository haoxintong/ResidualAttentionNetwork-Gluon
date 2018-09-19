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
"""Test script of Cifar-10 dataset."""

import os
import mxnet as mx
from tqdm import tqdm
from mxnet.gluon.data import DataLoader
from mxnet import image, nd, gluon, metric as mtc
from net.attention_net import get_attention_cifar

val_auglist = image.CreateAugmenter(data_shape=(3, 32, 32),
                                    mean=nd.array([0.485, 0.456, 0.406]),
                                    std=nd.array([0.229, 0.224, 0.225]))


def transform_val(im, label):
    im = im.astype('float32') / 255
    for aug in val_auglist:
        im = aug(im)
    im = nd.transpose(im, (2, 0, 1))
    return im, label


batch_size = 64
ctx = mx.gpu()

val_set = gluon.data.vision.CIFAR10(train=False, transform=transform_val)
val_data = DataLoader(val_set, batch_size, False, num_workers=2)

net = get_attention_cifar(10, num_layers=92)
net.load_parameters("models/cifar10-epoch-80.params", ctx=ctx)
net.hybridize()

metric = mtc.Accuracy()
cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
val_loss = 0

for i, batch in tqdm(enumerate(val_data)):
    data = batch[0].as_in_context(ctx)
    labels = batch[1].as_in_context(ctx)
    outputs = net(data)
    metric.update(labels, outputs)

_, val_acc = metric.get()
print(val_acc)
