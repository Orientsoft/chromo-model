start=$(date "+%Y-%m-%d-%H:%M:%S")
path=$(pwd)
CUDA_VISIBLE_DEVICES=1 python ../../../main.py \
    --lr 0.0001 \
    --lr_milestone 40 70 \
    --end_epoch 50 \
    --batch_size 16 \
    --root /home/voyager/data/chromosome/20191127 \
    --model resnet101 \
    --exp_path $path \
    --log_file $path/train-${start}.log \
    --log_freq 50 \
    --preprocess automask \
    --dataset ExtDataset \
    --num_classes 3 

    