# ReImplementation of Residual Attention Network for Image Classification
This is a Gluon implementation of the residual attention network in the paper [1704.06904](https://arxiv.org/abs/1704.06904).

## Requirement
python3.5, mxnet-1.2.1+, MXBoard

## Inspiration
The code is inspired by the gluon resnet implementation and https://github.com/liudaizong/Residual-Attention-Network.

## Train
### Cifar10
**Download Dataset**

```shell
wget http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
```

**Reorganize data**  
The command below will save the images to 10 dirs. Then we could use `ImageFolderDataset` to load the dataset.

```shell
python ./utils/reorg_cifar10_data.py --data-root /path/to/uncompressed/data --output-root /path/to/train/test/data
```

**Training**   

```shell
python ./train.py --data-root /path/to/train/test/data --workers 4 --num-gpus 2 --iterations 160000 
```

Most parameters is set to keep same with settings in paper. To view the training process, tensorboard is required.
 
```shell
tensorboard --logdir=./log/board/cifar10_20180831183421 --host=0.0.0.0 --port=8888
```

## TODO
- [x] train scripts
- [ ] attention module with other basic network unit  
...

## References
1. Residual Attention Network for Image Classification [1704.06904](https://arxiv.org/abs/1704.06904)
1. MXNet Documentation and Tutorials [zh.gluon.ai/](http://zh.gluon.ai/)