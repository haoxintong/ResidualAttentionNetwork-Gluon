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
from datetime import datetime
from mxnet import image, nd, gluon, metric as mtc, autograd as ag
from mxboard import SummaryWriter
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision import ImageFolderDataset
from net.attention_net import AttentionNet56Cifar


def parse_args():
    parser = argparse.ArgumentParser(description='Train AttentionNet56Cifar Net',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data-root', required=True, type=str,
                        help='root dir of the cifar-10 dataset, it should contain train and test')
    parser.add_argument("--log-dir", type=str, default="./log",
                        help="dataset; wiki or imdb")
    parser.add_argument('-j', '--workers', dest='num_workers', default=4, type=int,
                        help='number of preprocessing workers')
    parser.add_argument('--num-gpus', default=2, type=int,
                        help='number of gpus to use, 0 indicates cpu only')
    parser.add_argument('--iterations', default=160000, type=int,
                        help='number of training epochs')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        help='mini-batch size')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', dest='wd', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--lr-steps', default='64000,96000', type=str,
                        help='list of learning rate decay steps as in str')
    args = parser.parse_args()
    return args


rand_mirror = image.HorizontalFlipAug(0.5)
mean = nd.array([0.4914, 0.4822, 0.4465])
norm = image.ColorNormalizeAug(mean, None)


def transform_train(data, label):
    im = data.astype('float32') / 255
    im = norm(im)
    im = image.copyMakeBorder(im, 4, 4, 4, 4)
    im, _ = image.random_crop(im, (32, 32))
    im = rand_mirror(im)
    im = nd.transpose(im, (2, 0, 1))
    return im, label


def transform_val(data, label):
    im = data.astype('float32') / 255
    im = norm(im)
    im = nd.transpose(im, (2, 0, 1))
    return im, label


def train(args):
    # load_data
    train_set = ImageFolderDataset(os.path.join(args.data_root, "train"), transform=transform_train)
    train_data = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                            last_batch='discard')
    val_set = ImageFolderDataset(os.path.join(args.data_root, "test"), transform=transform_val)
    val_data = DataLoader(val_set, args.batch_size, False, num_workers=args.num_workers)

    # set the network and trainer
    ctx = [mx.gpu(i) for i in range(args.num_gpus)] if args.num_gpus > 0 else [mx.cpu()]
    net = AttentionNet56Cifar(10)
    net.initialize(init=mx.initializer.MSRAPrelu(), ctx=ctx)
    net.hybridize()

    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': args.lr, 'momentum': args.momentum, 'wd': args.wd})
    cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

    metric = mtc.Accuracy()

    # set log output
    logging.basicConfig(level=logging.INFO,
                        handlers=[
                            logging.StreamHandler(),
                            logging.FileHandler(os.path.join(args.log_dir, 'text/cifar10_%s.log')
                                                % datetime.strftime(datetime.now(), '%Y%m%d%H%M%S'))
                        ])
    sw = SummaryWriter(logdir=os.path.join(args.log_dir, 'board/cifar10_%s'
                                           % datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')))

    step = 0
    lr_counter = 0
    lr_steps = [64000, 96000]
    num_batch = len(train_data)
    while step < 160000:

        train_loss = 0
        metric.reset()
        tic = time.time()
        for i, batch in enumerate(train_data):
            if step == lr_steps[lr_counter]:
                trainer.set_learning_rate(trainer.learning_rate * 0.1)
                if lr_counter + 1 < len(lr_steps):
                    lr_counter += 1

            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
            labels = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)

            with ag.record():
                outputs = [net(X) for X in data]
                losses = [cross_entropy(yhat, y) for yhat, y in zip(outputs, labels)]
            for l in losses:
                ag.backward(l)

            trainer.step(args.batch_size)
            metric.update(labels, outputs)

            batch_loss = sum([l.mean().asscalar() for l in losses]) / len(losses)
            train_loss += batch_loss

            _, batch_acc = metric.get()
            sw.add_scalar("Loss", ('train', batch_loss), step)

            if (step % 20000) == 0 and step != 0:
                net.save_parameters("./models/cifar10_iter%d.params" % step)
            step += 1

        _, train_acc = metric.get()
        train_loss /= num_batch
        val_acc, val_loss = validate(net, val_data, ctx)

        sw.add_scalar("Loss", ('val', val_loss), step)
        sw.add_scalar("Accuracy", {'train': train_acc, 'val': val_acc}, step)
        logging.info('[Iteration %d] train accuracy: %.3f, train loss: %.3f | '
                     'val accuracy: %.3f, val loss: %.3f, time: %.1f'
                     % (step, train_acc, train_loss, val_acc, val_loss, time.time() - tic))

    net.export("./models/cifar10-model")
    sw.close()
    logging.info("Train End.")


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
    train(parse_args())
