start=$(date "+%Y-%m-%d-%H:%M:%S")
path=$(pwd)
# specified the val phase as 'train' in main.py will eval the trainset
CUDA_VISIBLE_DEVICES=0 python ../../../main.py \
    --lr 0.0001 \
    --lr_milestone 30 50 \
    --end_epoch 50 \
    --batch_size 16 \
    --model resnet101 \
    --val_list /home/voyager/data/chromosome/20191127/list/val.txt \
    --save_val_result "" \
    --exp_path $path \
    --log_file $path/train-${start}.log \
    --log_freq 50 \
    --preprocess automask \
    --num_classes 3 \
    --resume \
    --save_output \
    --checkpoint /home/voyager/jpz/chromosome/exp/ext/res101_bs16_epoch80_automask_lr0.0001/checkpoint/epoch_0037_acc_71.444099.pth \
    --eval