#!/bin/bash

if [ ! -d "ab_crown_proxy" ]; then
    mkdir "ab_crown_proxy"
fi

models_dir=$1
model_cnt=$2

for (( i=0; i<$model_cnt; i++ )); do
    echo $i
    cp "$models_dir/$i" "ab_crown_proxy/model"
    cp "$models_dir/${i}_predictions.npy" "ab_crown_proxy/labels"
    cd "../alpha-beta-CROWN/complete_verifier"
    python abcrown.py --config ../../defend_framework/cifar10_cnn_a_adv.yaml >/dev/null
    mv "Verified_ret_[cifar10_cnn_4layer_def]_start=0_end=10000_iter=20_b=1024_timeout=400_branching=kfsb-min-3_lra-init=0.1_lra=0.01_lrb=0.05_PGD=before_cplex_cuts=False_multiclass=allclass_domain.npy" "../../defend_framework/$models_dir/verified_${i}"
    cd "../../defend_framework"
done