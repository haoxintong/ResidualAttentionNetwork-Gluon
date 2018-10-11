python3 cifar10_train.py --num-layers 56 --num-gpus 2 --workers 6 --batch-size 64 --epochs 220 --lr-steps 80,120,200 --mix-up 1 --alpha 1.0
python3 cifar10_train.py --num-layers 92 --num-gpus 2 --workers 6 --batch-size 64 --epochs 220 --lr-steps 80,120,200 --mix-up 1 --alpha 1.0
python3 cifar10_train.py --num-layers 56 --num-gpus 2 --workers 6 --batch-size 64 --epochs 220 --lr-steps 80,120,200 --mix-up 0
python3 cifar10_train.py --num-layers 92 --num-gpus 2 --workers 6 --batch-size 64 --epochs 220 --lr-steps 80,120,200 --mix-up 0
