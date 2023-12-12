#! /bin/bash
declare -a array_datasets=(CLEVR_COUNT CLEVR_DISTANCE DMLAB KITTI_DISTANCE DS_LOC DS_ORI SN_AZI SN_ELE)
declare -a array_class=(8 6 6 4 16 16 18 9)
declare -a array_lr=(1e-3 1e-3 1e-3 1e-3 1e-3 1e-3 1e-3 1e-3)
for seed in 0 1 2 3 4 5 6 7; do
    python vit_train_sa2vp.py \
        --output_dir ./model_save/${array_datasets[${seed}]} \
        --update_freq 1 \
        --warmup_epochs 10 \
        --epochs 100 \
        --drop_path 0.0 \
        --log_dir ./log_save \
        --my_mode testall \
        --min_lr 1e-7 \
        --data_set ${array_datasets[${seed}]} \
        --nb_classes ${array_class[${seed}]} \
        --lr ${array_lr[${seed}]} \
        --weight_decay 1e-4 \
        --batch_size 40 
done