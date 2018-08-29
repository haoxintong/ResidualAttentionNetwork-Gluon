from mxnet.gluon import HybridBlock, nn
from mxnet.gluon.model_zoo.vision.resnet import BasicBlockV2, BottleneckV2


class AttentionBlock(HybridBlock):
    def __init__(self, channels, p=1, t=2, r=1, **kwargs):
        """

        :param p: the number of pre-processing Residual Units before split into trunk branch and mask branch.
        :param t: the number of Residual Units in trunk branch.
        :param r: the number of Residual Units between adjacent pooling layer in the mask branch.
        :param kwargs:
        """
        super().__init__(**kwargs)
        with self.name_scope():
            self.pre = nn.HybridSequential(prefix='pre_')
            for i in range(p):
                self.pre.add(BottleneckV2(channels, 1, prefix='%d_' % i))

            self.trunk_branch = nn.HybridSequential(prefix='trunk_')
            for i in range(t):
                self.trunk_branch.add(BottleneckV2(channels, 1, prefix='%d_' % i))

            self.mask_branch = MaskBlock(channels, r, prefix='mask_')

            self.post = nn.HybridSequential(prefix='post_')

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.pre(x)
        mask = self.mask_branch(x)
        trunk = self.trunk_branch(x)
        out = (1 + mask) * trunk
        out = self.post(out)
        return out


class UpSampleBlock(HybridBlock):
    def __init__(self, out_size, **kwargs):
        super().__init__(**kwargs)
        self._size = out_size

    def hybrid_forward(self, F, x, *args, **kwargs):
        return F.contrib.BilinearResize2D(x, height=self._size, width=self._size)


class MaskBlock(HybridBlock):
    def __init__(self, channels, r, **kwargs):
        super().__init__(**kwargs)

        with self.name_scope():
            self.down_sample_1 = nn.MaxPool2D(3, 2, 1)
            self.down_res_unit_1 = nn.HybridSequential(prefix="down_res1_")
            for i in range(r):
                self.down_res_unit_1.add(BottleneckV2(channels, 1, prefix="%d_" % i))
            self.skip_connection_1 = BottleneckV2(channels, 1)

            self.down_sample_2 = nn.MaxPool2D(3, 2, 1)
            self.down_res_unit_2 = nn.HybridSequential(prefix="down_res2_")
            for i in range(r):
                self.down_res_unit_2.add(BottleneckV2(channels, 1, prefix="%d_" % i))
            self.skip_connection_2 = BottleneckV2(channels, 1)

            self.down_sample_3 = nn.MaxPool2D(3, 2, 1)
            self.down_res_unit_3 = nn.HybridSequential(prefix="down_res3_")
            for i in range(r):
                self.down_res_unit_3.add(BottleneckV2(channels, 1, prefix="%d_" % i))

            self.up_res_unit_3 = nn.HybridSequential(prefix="up_res3_")
            for i in range(r):
                self.up_res_unit_3.add(BottleneckV2(channels, 1, prefix="%d_" % i))
            self.up_sample_3 = UpSampleBlock()

            self.up_res_unit_2 = nn.HybridSequential(prefix="up_res2_")
            for i in range(r):
                self.up_res_unit_2.add(BottleneckV2(channels, 1, prefix="%d_" % i))
            self.up_sample_2 = UpSampleBlock()

            self.up_res_unit_1 = nn.HybridSequential(prefix="up_res1_")
            for i in range(r):
                self.up_res_unit_1.add(BottleneckV2(channels, 1, prefix="%d_" % i))
            self.up_sample_1 = UpSampleBlock()

            self.output = nn.HybridSequential(prefix="output_")
            self.output.add(nn.BatchNorm(),
                            nn.Activation('relu'),
                            nn.Conv2D(channels, kernel_size=1, strides=1, use_bias=False),
                            nn.BatchNorm(),
                            nn.Activation('relu'),
                            nn.Conv2D(channels, kernel_size=1, strides=1, use_bias=False),
                            nn.Activation('sigmoid')
                            )

    def hybrid_forward(self, F, x, *args, **kwargs):
        x_down1 = self.down_sample_1(x)
        x_down1 = self.down_res_unit_1(x_down1)
        residual_1 = self.skip_connection_1(x_down1)

        x_down2 = self.down_sample_2(x_down1)
        x_down2 = self.down_res_unit_2(x_down2)
        residual_2 = self.skip_connection_2(x_down2)

        x_down3 = self.down_sample_3(x_down2)
        x_down3 = self.down_res_unit_3(x_down3)

        x_up3 = self.up_res_unit_3(x_down3)
        x_up3 = self.up_sample_3(x_up3)

        x_up2 = x_up3 + residual_2
        x_up2 = self.up_res_unit_2(x_up2)
        x_up2 = self.up_sample_2(x_up2)

        x_up1 = x_up2 + residual_1
        x_up1 = self.up_res_unit_1(x_up1)
        x_up1 = self.up_sample_1(x_up1)

        out = self.output(x_up1)
        return out
