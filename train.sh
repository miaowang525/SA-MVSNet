#!/usr/bin/env bash

# train on DTU's training set
MVS_TRAINING="/home/mvs_training/dtu/"

python train.py --batch_size 1 --epochs 60 --trainpath=/home/d6813/MW/PatchmatchNet-new/data/dtu2/ --trainlist lists/dtu/train.txt \
--vallist lists/dtu/val.txt --num_light_idx=3 "$@"
