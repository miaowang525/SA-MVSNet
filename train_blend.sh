#!/usr/bin/env bash

# train on DTU's training set
# --loadckpt=./checkpoints/convnext_tiny_1k_224_ema.pth
# --loadckpt=./checkpoints/32_25.ckpt
#python train.py --batch_size 3 --epochs 100 --trainpath=/home/d6813/MW/data/dtu2/ --trainlist lists/dtu/train.txt \
#--vallist lists/dtu/val.txt --num_light_idx=3 "$@"

python train_blend.py --batch_size 2 --epochs 100 --trainpath=/home/d6813/MW/data/blendedmvs/ --trainlist lists/blend/all.txt \
--vallist lists/blend/val.txt --num_light_idx=1 --resume"$@"

#CUDA_VISIBLE_DEVICES="2,3" python -m torch.distributed.launch --nproc_per_node 2 train111.py --batch_size=2 --epochs=100 --trainpath=/home/d6813/MW/data/dtu2/ --trainlist lists/dtu/train.txt \
#--vallist lists/dtu/val.txt --num_light_idx=3 "$@"
