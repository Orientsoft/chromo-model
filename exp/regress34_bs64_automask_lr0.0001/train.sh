start=$(date "+%Y-%m-%d-%H:%M:%S")
path=$(pwd)
CUDA_VISIBLE_DEVICES=0 python ../../main.py \
    --lr 0.0001 \
    --end_epoch 50 \
    --batch_size 64 \
    --root /home/voyager/data/chromosome/ \
    --model regress_resnet34 \
    --exp_path $path \
    --log_file $path/train-${start}.log \
    --log_freq 50 \
    --preprocess automask \
    --resume \
    --checkpoint $path/checkpoint/epoch_0013_perc_61.513158.pth