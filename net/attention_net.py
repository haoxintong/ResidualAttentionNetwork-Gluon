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
"""Residual Attention network, implemented in Gluon."""

from mxnet.gluon import nn
from .attention_block import BottleneckV2, AttentionBlock

__all__ = ["AttentionNet56", "AttentionNet92", "AttentionNet56Cifar"]


class AttentionNet56(nn.HybridBlock):
    r"""AttentionNet 56 Model from
    `"Residual Attention Network for Image Classification"
    <https://arxiv.org/abs/1704.06904>`_ paper.

    Parameters
    ----------
    :param classes: int. Number of classification classes.
    :param kwargs:

    """

    def __init__(self, classes, **kwargs):
        super().__init__(**kwargs)
        with self.name_scope():
            self.features = nn.HybridSequential()
            # 112x112
            self.features.add(nn.Conv2D(64, 7, 2, 3, use_bias=False))
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))

            # 56x56
            self.features.add(nn.MaxPool2D(3, 2, 1))
            self.features.add(BottleneckV2(256, 1, True, 64),
                              AttentionBlock(256, 56, stage=1, p=1, t=2, r=1))

            # 28x28
            self.features.add(BottleneckV2(512, 2, True, 256),
                              AttentionBlock(512, 28, stage=2, p=1, t=2, r=1))

            # 14x14
            self.features.add(BottleneckV2(1024, 2, True, 512),
                              AttentionBlock(1024, 14, stage=3, p=1, t=2, r=1))

            # 7x7
            self.features.add(BottleneckV2(2048, 2, True, 1024),
                              BottleneckV2(2048, 1),
                              BottleneckV2(2048, 1))

            # 2048
            self.features.add(nn.BatchNorm(),
                              nn.Activation('relu'),
                              nn.GlobalAvgPool2D(),
                              nn.Flatten())

            # classes
            self.output = nn.Dense(classes)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.features(x)
        x = self.output(x)
        return x


class AttentionNet92(nn.HybridBlock):
    r"""AttentionNet 92 Model from
    `"Residual Attention Network for Image Classification"
    <https://arxiv.org/abs/1704.06904>`_ paper.

    Parameters
    ----------
    :param classes: int. Number of classification classes.
    :param kwargs:

    """

    def __init__(self, classes, **kwargs):
        super().__init__(**kwargs)
        with self.name_scope():
            self.features = nn.HybridSequential()
            # 112x112
            self.features.add(nn.Conv2D(64, 7, 2, 3, use_bias=False))
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))

            # 56x56
            self.features.add(nn.MaxPool2D(3, 2, 1))
            self.features.add(BottleneckV2(256, 1, True, 64),
                              AttentionBlock(256, 56, stage=1, p=1, t=2, r=1))

            # 28x28
            self.features.add(BottleneckV2(512, 2, True, 256),
                              AttentionBlock(512, 28, stage=2, p=1, t=2, r=1),
                              AttentionBlock(512, 28, stage=2, p=1, t=2, r=1))

            # 14x14
            self.features.add(BottleneckV2(1024, 2, True, 512),
                              AttentionBlock(1024, 14, stage=3, p=1, t=2, r=1),
                              AttentionBlock(1024, 14, stage=3, p=1, t=2, r=1),
                              AttentionBlock(1024, 14, stage=3, p=1, t=2, r=1))

            # 7x7
            self.features.add(BottleneckV2(2048, 2, True, 1024),
                              BottleneckV2(2048, 1),
                              BottleneckV2(2048, 1))

            # 2048
            self.features.add(nn.BatchNorm(),
                              nn.Activation('relu'),
                              nn.GlobalAvgPool2D(),
                              nn.Flatten())

            # classes
            self.output = nn.Dense(classes)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.features(x)
        x = self.output(x)
        return x


class AttentionNet56Cifar(nn.HybridBlock):
    r"""
    AttentionNet 56 Model for input 32x32.

    Parameters
    ----------
    :param classes: int. Number of classification classes.
    :param kwargs:

    """

    def __init__(self, classes, **kwargs):
        super().__init__(**kwargs)
        with self.name_scope():
            self.features = nn.HybridSequential()
            # 32x32
            self.features.add(nn.Conv2D(32, 5, 1, 2, use_bias=False))
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))

            # 16x16
            self.features.add(nn.MaxPool2D(2, 2, 0))
            self.features.add(BottleneckV2(128, 1, True, 32),
                              AttentionBlock(128, 16, stage=1, p=1, t=2, r=1))

            # 8x8
            self.features.add(BottleneckV2(256, 2, True, 128),
                              AttentionBlock(256, 8, stage=2, p=1, t=2, r=1))

            # 4x4
            self.features.add(BottleneckV2(512, 2, True, 256),
                              AttentionBlock(512, 4, stage=3, p=1, t=2, r=1))

            # 4x4
            self.features.add(BottleneckV2(1024, 2, True, 512),
                              BottleneckV2(1024, 1),
                              BottleneckV2(1024, 1))

            # 2048
            self.features.add(nn.BatchNorm(),
                              nn.Activation('relu'),
                              nn.GlobalAvgPool2D(),
                              nn.Flatten())

            # classes
            self.output = nn.Dense(classes)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.features(x)
        x = self.output(x)
        return x
