#! /bin/bash
checkpoint=/home/voyager/jpz/chromosome/exp/ext/res101_bs16_epoch80_automask_lr0.0001/checkpoint # 将该文件夹下的所有pth都进行eval
path=$(pwd)
start=$(date "+%Y-%m-%d-%H:%M:%S")
logfile=$path/autoeval-${start}.log # log保存在当前目录下
for file in `ls $checkpoint`
do
    echo $checkpoint/$file
    sufix=${file##*.}
    if [ $sufix = 'pth' ];then
        CUDA_VISIBLE_DEVICES=0 python ../../../main.py \
            --batch_size 16 \
            --val_list /home/voyager/data/chromosome/20191127/list/val.txt \
            --model resnet101 \
            --exp_path $path \
            --log_file $logfile \
            --log_freq 50 \
            --preprocess automask \
            --num_classes 3 \
            --resume \
            --checkpoint  $checkpoint/$file \
            --eval
    fi
done