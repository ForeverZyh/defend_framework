#!/bin/bash

friendly_lrs=(
  100
  50
  20
  10
)

friendly_mus=(
  1
  10
)

for friendly_lr in "${friendly_lrs[@]}"; do
  for friendly_mu in "${friendly_mus[@]}"; do
        log_file="result/log_friendly_lr_${friendly_lr}_mu_${friendly_mu}_cifar10-02.txt"
    #    log_file="result/log_friendly_lr_${friendly_lr}_mu_${friendly_mu}_mnist.txt"
#    log_file="result/log_friendly_lr_${friendly_lr}_mu_${friendly_mu}_ember.txt"
    rm ${log_file} -f
    for seed in {0..4}; do
            CUDA_VISIBLE_DEVICES=0 python friendly_noise_train.py \
              --load_poison_dir /nobackup/yuhao_data/backdoored_data/attack_cifar10_02_0_002_oneside --dataset cifar10-02 \
              --epochs 40 --nesterov --seed ${seed} --noise_type friendly gaussian --friendly_begin_epoch 5 --no-progress \
              --friendly_lr ${friendly_lr} --friendly_mu ${friendly_mu} --val_freq 100 >>${log_file} 2>&1
      #      CUDA_VISIBLE_DEVICES=1 python friendly_noise_train.py \
      #        --load_poison_dir /nobackup/yuhao_data/backdoored_data/attack_mnist_0_002_4 --dataset mnist \
      #        --epochs 40 --nesterov --seed ${seed} --noise_type friendly gaussian --friendly_begin_epoch 5 \
      #        --noise_eps 8 --friendly_clamp 8 --no-progress \
      #        --friendly_lr ${friendly_lr} --friendly_mu ${friendly_mu} --val_freq 100 >>${log_file} 2>&1
#      CUDA_VISIBLE_DEVICES=0 python friendly_noise_train.py \
#        --load_poison_dir /nobackup/yuhao_data/malware_poison/embernn_fig2_3_0.001_nd_na/ember__embernn__combined_shap__combined_shap__all \
#        --ember_data_dir /nobackup/yuhao_data/malware_poison/data/ember --dataset ember \
#        --noise_type gaussian --friendly_begin_epoch 1 --epochs 10 --batch-size 512 --nesterov --seed ${seed} \
#        --noise_eps 32 --friendly_clamp 32 --no-progress --val_freq 100 \
#        --friendly_lr ${friendly_lr} --friendly_mu ${friendly_mu} >>${log_file} 2>&1
    done
  done
done
