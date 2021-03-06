start=$(date "+%Y-%m-%d-%H:%M:%S")
path=$(pwd)
CUDA_VISIBLE_DEVICES=1 python ../../main.py \
    --lr 0.0001 \
    --end_epoch 50 \
    --batch_size 64 \
    --root /home/voyager/data/chromosome/ \
    --model bi18 \
    --exp_path $path \
    --log_file $path/train-${start}.log \
    --log_freq 50 \
    --preprocess automask

    #--resume \
    #--checkpoint $path/checkpoint/epoch_0004_perc_89.571974.pth