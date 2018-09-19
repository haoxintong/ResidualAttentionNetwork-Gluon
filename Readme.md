# ReImplementation of Residual Attention Network for Image Classification
This is a Gluon implementation of the residual attention network in the paper [1704.06904](https://arxiv.org/abs/1704.06904).

<img src="data/figure2.png"/>

## Requirement
python3.5+, mxnet-1.2.1+, MXBoard

## Inspiration
The code is inspired by the gluon resnet implementation and https://github.com/liudaizong/Residual-Attention-Network.

## Train
GPU is preferred.
### Cifar10
To view the training process, tensorboard is required.
 
```shell
tensorboard --logdir=./log/board/cifar10_201808311834 --host=0.0.0.0 --port=8888
```

|Results|Accuracy|Loss |Test Accuracy|
|:---:  |:---:   |:---:|:---:       |
|Attention56|<img src="data/attention56-cifar10-accuracy.png"/>|<img src="data/attention56-cifar10-loss.png"/>|0.9310|
|Attention92|<img src="data/cifar10-attention92-accuracy.png"/>|<img src="data/cifar10-attention92-loss.png"/>|0.9524|
   
The test error reported in paper is **4.99%** for `Attention-92`, here I got **4.76%**. The author does not give the 
architecture of the cifar10-AttentionNet, so I follow the implementation of https://github.com/tengshaofeng/ResidualAttentionNetwork-pytorch.
In previous version, the feature map is down sampling to 16x16 before stage1, it can only achieve about 0.93 on test set. 
Following Teng's implementation, before stage1 the feature map size is still 32x32, and it got an accuracy improvement of 0.02.

 
### ImageNet
Emmmm....

## TODO
- [x] Training scripts for cifar10.  
Just use this command and you can get accuracy over 0.95 on cifar10:  
```shell
python3 cifar10_train.py --num-layers 92 --num-gpus 1 --workers 2 --batch-size 64 --epochs 200 --lr-steps 80,120
```
It can be easily applied to other tasks.

- [x] Attention-56,92,128,164,236,452 support  
The hyper-parameters are based on paper section 4.1. The number of layers can be calculated by 36m+20 
where m is the number of Attention Module in each stage when `p, t, r = 1, 2, 1`.
```python
attention_net_spec = {56: ([1, 2, 1], [1, 1, 1]),
                      92: ([1, 2, 1], [1, 2, 3]),
                      128: ([1, 2, 1], [3, 3, 3]),
                      164: ([1, 2, 1], [4, 4, 4]),
                      236: ([1, 2, 1], [6, 6, 6]),
                      452: ([2, 4, 3], [6, 6, 6])}
``` 
- [ ] Visualization of soft attention mask 
- [ ] Attention module with other basic network unit  
...

## References
1. Residual Attention Network for Image Classification [1704.06904](https://arxiv.org/abs/1704.06904)
1. MXNet Documentation and Tutorials [zh.gluon.ai/](http://zh.gluon.ai/)