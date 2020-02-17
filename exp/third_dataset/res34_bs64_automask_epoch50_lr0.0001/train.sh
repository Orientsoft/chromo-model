start=$(date "+%Y-%m-%d-%H:%M:%S")
path=$(pwd)
CUDA_VISIBLE_DEVICES=0 python ../../../main.py \
    --lr 0.0001 \
    --lr_milestones 30 \
    --end_epoch 80 \
    --batch_size 64 \
    --train_list /home/voyager/data/chromosome/20191204/list/train.txt \
    --val_list /home/voyager/data/chromosome/20191204/list/val.txt \
    --model resnet34 \
    --exp_path $path \
    --log_file $path/train-${start}.log \
    --log_freq 50 \
    --eval_freq 5 \
    --preprocess automask \
    --num_classes 3 \
    --loss FocalLoss \
    --resume \
    --checkpoint /home/voyager/jpz/chromosome/exp/third_dataset/res34_bs64_automask_epoch50_lr0.0001/checkpoint/epoch_0045_acc_81.929348.pth