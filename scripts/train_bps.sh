export NVIDIA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DMLC_WORKER_ID=0
export DMLC_NUM_WORKER=1
export DMLC_ROLE=worker

# the following value does not matter for non-distributed jobs
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=127.0.0.1
export DMLC_PS_ROOT_PORT=9000
# export BYTEPS_LOG_LEVEL=DEBUG

path="`dirname $0`"

cd ..

python $path/launch.py \
    python $path/../train.py --num-layers 56 --workers 2 --batch-size 128 --epochs 200 --mix-up 1 \
    --alpha 1.0 --lr 1.6 --epsilon 0.0 --float16 0 --warmup-epochs 10
