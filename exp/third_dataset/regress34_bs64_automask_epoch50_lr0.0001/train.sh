start=$(date "+%Y-%m-%d-%H:%M:%S")
path=$(pwd)
CUDA_VISIBLE_DEVICES=1 python ../../../main.py \
    --lr 0.0001 \
    --lr_milestones 30 \
    --end_epoch 50 \
    --batch_size 64 \
    --train_list /home/voyager/data/chromosome/20191204/list/train.txt \
    --val_list /home/voyager/data/chromosome/20191204/list/val.txt \
    --model regress_resnet34 \
    --exp_path $path \
    --log_file $path/train-${start}.log \
    --log_freq 50 \
    --eval_freq 5 \
    --preprocess automask \
    --loss L1Loss 