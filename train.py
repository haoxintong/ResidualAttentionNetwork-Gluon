# MIT License
# Copyright (c) 2019 haoxintong
"""Train script of Cifar-10 dataset."""
import os
import time
import logging
import argparse

import mxnet as mx
import numpy as np
from datetime import datetime
from mxnet import nd, gluon, metric as mtc, autograd as ag

from gluoncv.utils.lr_scheduler import LRSequential, LRScheduler
from net.attention_net import get_attention_cifar
import byteps.mxnet as bps

import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.mxnet import DALIClassificationIterator


def parse_args():
    parser = argparse.ArgumentParser(description='Train AttentionNetCifar',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num-layers", type=int, default=56,
                        help="number of layers of the attention net")
    parser.add_argument("--log-dir", type=str, default="./logs",
                        help="where to save log")
    parser.add_argument('-j', '--workers', dest='num_workers', default=2, type=int,
                        help='number of preprocessing workers')
    parser.add_argument('--epochs', default=200, type=int,
                        help='number of training epochs')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        help='mini-batch size of each gpu')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', dest='wd', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--warmup-epochs', default=5, type=int,
                        help='warmup epochs')
    parser.add_argument('--mix-up', default=0, type=int,
                        help='if use mix-up method to train net, 0 for False')
    parser.add_argument('--alpha', default=1.0, type=float,
                        help='hyper param of mix-up')
    parser.add_argument('--epsilon', default=0.0, type=float,
                        help='hyper param of label smoothing.')
    parser.add_argument('--float16', default=0, type=int,
                        help='use fp16, default is 0.')
    parser.add_argument('--save-period', type=int, default=10,
                        help='period in epoch of model saving.')
    _args = parser.parse_args()
    return _args


class CifarPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, num_shards, shard_id, use_fp16=False, train=True,
                 root=os.path.expanduser('./data')):
        super().__init__(batch_size, num_threads, device_id, seed=12)

        part = "train" if train else "test"
        idx_files = [os.path.join(root, "cifar10_{}.idx").format(part)]
        rec_files = [os.path.join(root, "cifar10_{}.rec").format(part)]

        self.num_classes = 10
        self.image_size = (32, 32)
        self.size = 0
        self.train = train

        for idx_file in idx_files:
            with open(idx_file, "r") as f:
                self.size += len(list(f.readlines()))

        self._input = ops.MXNetReader(path=rec_files, index_path=idx_files, random_shuffle=True if train else False,
                                      num_shards=num_shards, shard_id=shard_id, seed=12,
                                      tensor_init_bytes=self.image_size[0] * self.image_size[1] * 8)
        self._decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)

        self._cmnp = ops.CropMirrorNormalize(device="gpu",
                                             output_dtype=types.FLOAT16 if use_fp16 else types.FLOAT,
                                             output_layout=types.NCHW,
                                             crop=self.image_size,
                                             image_type=types.RGB,
                                             mean=[123.675, 116.28, 103.53],
                                             std=[58.395, 57.12, 57.375])
        if train:
            self.padding = ops.Paste(device="gpu", fill_value=128, ratio=1.25)
            self.px = ops.Uniform(range=(0, 1))
            self.py = ops.Uniform(range=(0, 1))

        self._uniform = ops.Uniform(range=(0.7, 1.3))
        self._coin = ops.CoinFlip(probability=0.5)

    def define_graph(self):
        inputs, labels = self._input(name="Reader")
        images = self._decode(inputs)
        if self.train:
            images = self.padding(images, paste_x=self.px(), paste_y=self.py())
        images = self._cmnp(images, mirror=self._coin())
        return images, labels.gpu()


args = parse_args()
use_mix_up = True if args.mix_up else False
use_float16 = True if args.float16 else False
label_smoothing = args.epsilon > 0
epsilon = args.epsilon

bps.init()
num_gpu = bps.size()
local_rank = bps.local_rank()
rank = bps.rank()

epochs = args.epochs + 1
alpha = args.alpha
max_accuracy = 0.0

ctx = mx.gpu(bps.local_rank())

# load_data
batch_size = args.batch_size * num_gpu

train_pipes = [CifarPipe(args.batch_size, args.num_workers, local_rank, num_gpu, rank, use_float16)]
train_size = train_pipes[0].size
train_data = DALIClassificationIterator(train_pipes, train_size // num_gpu, auto_reset=True)
val_pipes = [CifarPipe(args.batch_size, args.num_workers, local_rank, 1, 0, use_float16, train=False)]
val_size = val_pipes[0].size
val_data = DALIClassificationIterator(val_pipes, val_size, auto_reset=True)

# set the network and trainer
net = get_attention_cifar(10, num_layers=args.num_layers)
net.initialize(init=mx.initializer.MSRAPrelu(), ctx=ctx)
net.hybridize()
if use_float16:
    net.cast("float16")

num_batches = train_size // batch_size
warm_up = LRScheduler(mode="linear", nepochs=args.warmup_epochs, base_lr=args.lr * 0.01,
                      target_lr=args.lr, iters_per_epoch=num_batches)
cosine = LRScheduler(mode="cosine", base_lr=args.lr, target_lr=1e-8,
                     iters_per_epoch=num_batches, nepochs=epochs, offset=args.warmup_epochs * num_batches)
lr_scheduler = LRSequential([warm_up, cosine])

params = net.collect_params()

trainer = bps.DistributedTrainer(params, 'nag',
                                 {'learning_rate': args.lr, 'momentum': args.momentum, 'wd': args.wd,
                                  "multi_precision": True, "lr_scheduler": lr_scheduler})

cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=not use_mix_up)

# set log output
train_mode = 'MixUP' if use_mix_up else 'Vanilla'
logger = logging.getLogger('TRAIN')
if rank == 0:
    logger.setLevel("INFO")
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(logging.FileHandler(os.path.join(args.log_dir, 'cifar10_attention%d_%s_%s.log'
                                                       % (args.num_layers, train_mode,
                                                          datetime.strftime(datetime.now(), '%Y%m%d%H%M')))))
    logger.info(args)
    logger.info("Batch size: {} Num GPU: {}".format(batch_size, num_gpu))

train_metric = mtc.Accuracy() if not use_mix_up else mx.metric.RMSE()
loss_metric = mx.metric.Loss()
val_metric = mtc.Accuracy()

for epoch in range(epochs):

    tic = time.time()
    for i, batch in enumerate(train_data):

        data = batch[0].data[0]
        label = batch[0].label[0]

        if use_mix_up and epoch < epochs * 0.9:
            cross_entropy._sparse_label = False
            label = nd.one_hot(label, depth=10, on_value=1 - epsilon, off_value=epsilon / (10 - 1),
                               dtype="float16" if use_float16 else "float32")
            lam = np.random.beta(alpha, alpha)

            data = lam * data + (1 - lam) * data[::-1]
            label = lam * label + (1 - lam) * label[::-1]

        elif label_smoothing:
            cross_entropy._sparse_label = False
            label = nd.one_hot(label, depth=10, on_value=1 - epsilon, off_value=epsilon / (10 - 1),
                               dtype="float16" if use_float16 else "float32")
        else:
            train_metric = mtc.Accuracy()
            train_metric.reset()
            cross_entropy._sparse_label = True

        with ag.record():
            output = net(data)
            loss = cross_entropy(output, label)

        ag.backward(loss)

        trainer.step(batch_size)
        train_metric.update([label], [output])
        loss_metric.update(0, [loss])

    btoc = time.time()
    _, train_acc = train_metric.get()
    _, train_loss = loss_metric.get()

    train_metric.reset()
    loss_metric.reset()

    if rank == 0:
        for i, batch in enumerate(val_data):
            data = batch[0].data[0]
            label = batch[0].label[0]

            output = net(data)

            cross_entropy._sparse_label = True
            val_metric.update([label], [output])
            loss_metric.update(0, [cross_entropy(output, label)])

        _, val_acc = val_metric.get()
        _, val_loss = loss_metric.get()
        max_accuracy = max(max_accuracy, val_acc)
        val_metric.reset()
        loss_metric.reset()

        toc = time.time()
        logger.info('[Epoch %d] train metric: %.6f, train loss: %.6f | '
                    'val accuracy: %.6f, val loss: %.6f, speed: %.2f samples/s | learning rate: %.8f,  time: %.1f'
                    % (epoch, train_acc, train_loss, val_acc, val_loss, train_size / (btoc - tic),
                       trainer.learning_rate, toc - tic))

    # if (epoch % args.save_period) == 0 and epoch != 0:
    #     net.save_parameters("./models/attention%d-cifar10-epoch-%d-%s.params"
    #                         % (args.num_layers, epoch, train_mode))
    #
    # if val_acc > max_accuracy:
    #     net.save_parameters("./models/best-%f-attention%d-cifar10-epoch-%d-%s.params"
    #                         % (val_acc, args.num_layers, epoch, train_mode))
    #     max_accuracy = val_acc

if rank == 0:
    logger.info("Finish training. Max accuracy: {:.6f}.".format(max_accuracy))
nd.waitall()
os._exit(0)
