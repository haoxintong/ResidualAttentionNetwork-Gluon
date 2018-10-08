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
"""Train script of Cifar-10 dataset."""

import os
import time
import logging
import argparse
import mxnet as mx
import numpy as np
from datetime import datetime
from mxnet import image, nd, gluon, metric as mtc, autograd as ag
from mxboard import SummaryWriter
from mxnet.gluon.data import DataLoader
from net.attention_net import get_attention_cifar


def parse_args():
    parser = argparse.ArgumentParser(description='Train AttentionNetCifar',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num-layers", type=int, default=56,
                        help="number of layers of the attention net")
    parser.add_argument("--log-dir", type=str, default="./log",
                        help="where to save log")
    parser.add_argument('-j', '--workers', dest='num_workers', default=4, type=int,
                        help='number of preprocessing workers')
    parser.add_argument('--num-gpus', default=2, type=int,
                        help='number of gpus to use, 0 indicates cpu only')
    parser.add_argument('--epochs', default=200, type=int,
                        help='number of training epochs')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        help='mini-batch size of each gpu')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', dest='wd', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--lr-steps', default='80,120', type=str,
                        help='list of learning rate decay steps as in str')
    parser.add_argument('--mix-up', default=False, type=bool,
                        help='if use mix-up method to train net')
    parser.add_argument('--alpha', default=1.0, type=float,
                        help='hyper param of mix-up')
    parser.add_argument('--save-period', type=int, default=10,
                        help='period in epoch of model saving.')
    _args = parser.parse_args()
    return _args


os.environ['MXNET_GLUON_REPO'] = 'https://apache-mxnet.s3.cn-north-1.amazonaws.com.cn/'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'

train_auglist = image.CreateAugmenter(data_shape=(3, 32, 32),
                                      rand_crop=True, rand_mirror=True,
                                      mean=nd.array([0.485, 0.456, 0.406]),
                                      std=nd.array([0.229, 0.224, 0.225]))
val_auglist = image.CreateAugmenter(data_shape=(3, 32, 32),
                                    mean=nd.array([0.485, 0.456, 0.406]),
                                    std=nd.array([0.229, 0.224, 0.225]))
args = parse_args()
use_mix_up = args.mix_up


def transform_train(data, label):
    im = data.astype('float32') / 255
    im = image.copyMakeBorder(im, 4, 4, 4, 4)
    for aug in train_auglist:
        im = aug(im)
    im = nd.transpose(im, (2, 0, 1))
    if use_mix_up:
        label = nd.one_hot(nd.array([label]), 10)[0]
    return im, label


def transform_val(data, label):
    im = data.astype('float32') / 255
    for aug in val_auglist:
        im = aug(im)
    im = nd.transpose(im, (2, 0, 1))
    return im, label


def train():
    # load_data
    batch_size = args.batch_size * max(args.num_gpus, 1)
    train_set = gluon.data.vision.CIFAR10(train=True, transform=transform_train)
    train_data = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=args.num_workers,
                            last_batch='discard')
    val_set = gluon.data.vision.CIFAR10(train=False, transform=transform_val)
    val_data = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

    # set the network and trainer
    ctx = [mx.gpu(i) for i in range(args.num_gpus)] if args.num_gpus > 0 else [mx.cpu()]
    net = get_attention_cifar(10, num_layers=args.num_layers)
    net.initialize(init=mx.initializer.MSRAPrelu(), ctx=ctx)
    net.hybridize()

    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': args.lr, 'momentum': args.momentum, 'wd': args.wd})
    cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=not use_mix_up)
    train_metric = mtc.Accuracy() if not use_mix_up else mx.metric.RMSE()

    # set log output
    logger = logging.getLogger('TRAIN')
    logger.setLevel("INFO")
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(logging.FileHandler(os.path.join(args.log_dir, 'text/cifar10_%s.log')
                                          % datetime.strftime(datetime.now(), '%Y%m%d%H%M')))
    sw = SummaryWriter(logdir=os.path.join(args.log_dir, 'board/cifar10_%s'
                                           % datetime.strftime(datetime.now(), '%Y%m%d%H%M')), verbose=False)

    # record the training hyper parameters
    logger.info(args)
    lr_counter = 0
    lr_steps = [int(s) for s in args.lr_steps.strip().split(',')]
    num_batch = len(train_data)
    epochs = args.epochs + 1
    alpha = args.alpha

    for epoch in range(epochs):
        if epoch == lr_steps[lr_counter]:
            trainer.set_learning_rate(trainer.learning_rate * 0.1)
            if lr_counter + 1 < len(lr_steps):
                lr_counter += 1
        train_loss = 0
        train_metric.reset()
        tic = time.time()
        for i, batch in enumerate(train_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
            labels = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)

            if use_mix_up:
                lam = np.random.beta(alpha, alpha)
                if epoch >= epochs - 30:
                    lam = 1
                data = [lam * X + (1 - lam) * X[::-1] for X in data]
                labels = [lam * Y + (1 - lam) * Y[::-1] for Y in labels]

            with ag.record():
                outputs = [net(X) for X in data]
                losses = [cross_entropy(yhat, y) for yhat, y in zip(outputs, labels)]

            for l in losses:
                ag.backward(l)

            trainer.step(batch_size)
            train_metric.update(labels, outputs)

            train_loss += sum([l.mean().asscalar() for l in losses]) / len(losses)

        _, train_acc = train_metric.get()
        train_loss /= num_batch
        val_acc, val_loss = validate(net, val_data, ctx)
        sw.add_scalar("AttentionNet/Loss", {'train': train_loss, 'val': val_loss}, epoch)

        sw.add_scalar("AttentionNet/Metric", {'train': train_acc, 'val': val_acc}, epoch)
        logger.info('[Epoch %d] train metric: %.6f, train loss: %.6f | '
                    'val accuracy: %.6f, val loss: %.6f, time: %.1f'
                    % (epoch, train_acc, train_loss, val_acc, val_loss, time.time() - tic))

        if (epoch % args.save_period) == 0 and epoch != 0:
            net.save_parameters("./models/attention%d-cifar10-epoch-%d.params" % (args.num_layers, epoch))

    net.export("./models/attention%d-cifar10-%s" % (args.num_layers, datetime.strftime(datetime.now(), '%Y%m%d%H%M')))
    sw.close()
    logger.info("Train End.")


def validate(net, val_data, ctx):
    metric = mtc.Accuracy()
    cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
    val_loss = 0

    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        labels = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        outputs = [net(X) for X in data]
        loss = [cross_entropy(yhat, y) for yhat, y in zip(outputs, labels)]
        metric.update(labels, outputs)
        val_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)

    _, val_acc = metric.get()
    return val_acc, val_loss / len(val_data)


if __name__ == '__main__':
    train()
