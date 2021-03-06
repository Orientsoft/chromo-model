#! /bin/bash
checkpoint=/home/voyager/jpz/chromosome/exp/res34_bs64_epoch50_automask_lr0.0001/checkpoint # 将该文件夹下的所有pth都进行eval
path=$(pwd)
start=$(date "+%Y-%m-%d-%H:%M:%S")
logfile=$path/eval-${start}.log # log保存在当前目录下
for file in `ls $checkpoint`
do
    echo $checkpoint/$file
    sufix=${file##*.}
    if [ $sufix = 'pth' ];then
        CUDA_VISIBLE_DEVICES=0 python ../../main.py \
            --batch_size 64 \
            --root /home/voyager/data/chromosome/ \
            --model oldresnet34 \
            --exp_path $path \
            --log_file $logfile \
            --log_freq 50 \
            --eval \
            --resume \
            --preprocess automask \
            --eval \
            --checkpoint $checkpoint/$file
    fi
done