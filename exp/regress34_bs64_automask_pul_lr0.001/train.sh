start=$(date "+%Y-%m-%d-%H:%M:%S")
path=$(pwd)
CUDA_VISIBLE_DEVICES=0 python ../../main.py \
    --lr 0.001 \
    --lr_milestones 25 35 \
    --end_epoch 50 \
    --batch_size 64 \
    --root /home/voyager/data/chromosome/ \
    --model pu_regress_resnet34 \
    --exp_path $path \
    --log_file $path/train-${start}.log \
    --log_freq 50 \
    --preprocess automask \
    --loss PULoss \
    --dataset PUDataset