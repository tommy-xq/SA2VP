#! /bin/bash
declare -a array_datasets=(CIFAR CAL101 DTD FLOWER_S PET SVHN_S SUN)
declare -a array_class=(100 102 47 102 37 10 397)
declare -a array_lr=(1e-3 1e-3 1e-3 1e-3 1e-3 1e-3 1e-3)
for seed in 0 1 2 3 4 5 6; do
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
        --batch_size 64 
done