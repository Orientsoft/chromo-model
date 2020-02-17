start=$(date "+%Y-%m-%d-%H:%M:%S")
path=$(pwd)
CUDA_VISIBLE_DEVICES=0 python ../../main.py \
    --lr 0.0001 \
    --end_epoch 200 \
    --batch_size 64 \
    --root /home/voyager/data/chromosome/ \
    --model regress_resnet34 \
    --exp_path $path \
    --log_file $path/eval-${start}.log \
    --log_freq 50 \
    --eval \
    --resume \
    --preprocess automask \
    --eval \
    --checkpoint /home/voyager/jpz/chromosome/exp/regress34_bs64_automask_lr0.0001/checkpoint/epoch_0013_perc_61.513158.pth
    
    #--checkpoint /home/voyager/jpz/chromosome/exp/res34_bs64_epoch50_automask_lr0.0001/checkpoint/epoch_0004_perc_87.292121.pth

    #--checkpoint /home/voyager/jpz/chromosome-server/data/classify_state_dict.pth