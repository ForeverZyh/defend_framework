#!/bin/bash

noise_eps=$1
friendly_clamp=$2

if [ -z "$noise_eps" ] || [ -z "$friendly_clamp" ]; then
  echo "Usage: $0 noise_eps friendly_clamp"
  exit 1
fi

log_file="result/log_${noise_eps}_${friendly_clamp}.out"

for friendly_begin_epoch in {2..4}
do
  for seed in {0..4}
  do
    python friendly_noise_train.py \
    --noise_type friendly bernoulli \
    --friendly_begin_epoch ${friendly_begin_epoch} \
    --epochs 10 \
    --out result \
    --load_poison_dir /nobackup/yuhao_data/malware_poison/embernn_fig2_3_0.001_nd_na/ember__embernn__combined_shap__combined_shap__all \
    --ember_data_dir /nobackup/yuhao_data/malware_poison/data/ember \
    --friendly_epochs 30 \
    --batch-size 512 \
    --val_freq=10 \
    --lr 0.1 \
    --noise_eps=${noise_eps} \
    --friendly_clamp=${friendly_clamp} \
    --seed ${seed} \
    >> ${log_file} 2>&1
  done
done
