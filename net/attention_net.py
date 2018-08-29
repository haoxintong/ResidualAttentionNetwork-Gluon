import mxnet as mx
from mxnet import gluon
from mxnet.gluon import HybridBlock, nn
from mxnet.gluon.model_zoo.vision.resnet import BottleneckV2
from net.attention_block import AttentionBlock


class AttentionNet56(HybridBlock):
    r"""ResAttentionNet 56
    `"Residual Attention Network for Image Classification"
    <https://arxiv.org/abs/1704.06904>`_ paper.

    Parameters
    ----------


    """

    def __init__(self, classes, **kwargs):
        super().__init__(**kwargs)
        with self.name_scope():
            self.features = gluon.nn.HybridSequential()
            # 112x112
            self.features.add(nn.Conv2D(64, 7, 2, 3, use_bias=False))
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))

            # 56x56
            self.features.add(nn.MaxPool2D(3, 2, 1))
            self.features.add(BottleneckV2(256, 1, True, 64),
                              AttentionBlock(256))

            # 28x28
            self.features.add(BottleneckV2(512, 2, True, 256),
                              AttentionBlock(512))

            # 14x14
            self.features.add(BottleneckV2(1024, 2, True, 512),
                              AttentionBlock(1024))

            # 7x7
            self.features.add(BottleneckV2(2048, 2, True, 1024),
                              BottleneckV2(2048, 1),
                              BottleneckV2(2048, 1))

            # 2048
            self.features.add(nn.GlobalAvgPool2D(),
                              nn.Flatten())

            # classes
            self.output = nn.Dense(classes)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.features(x)
        x = self.output(x)
        return x


if __name__ == '__main__':
    xi = mx.nd.ones(shape=(1, 3, 224, 224))
    net = AttentionNet56(2)
    net.initialize()
    y = net(xi)
    print(y.shape)
