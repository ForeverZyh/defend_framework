"""
This file runs in the conda environment of alpha-beta crown.
It assumes ../alpha-beta-crown is the directory of alpha-beta crown.
It runs at the directory of defend_framework.
python ab_crown_proxy.py <models_dir> <model_cnt> <exp_dir>
"""
import os
import sys
import pickle
import numpy as np

if __name__ == "__main__":
    if not os.path.exists("ab_crown_proxy"):
        os.mkdir("ab_crown_proxy")
    models_dir = sys.argv[1]
    model_cnt = int(sys.argv[2])
    exp_dir = sys.argv[3]
    res, _ = np.load(os.path.join(exp_dir, "aggre_res.npy"))
    noise_res = np.zeros_like(res)
    noise_res[:, -1] = res[:, -1]
    for i in range(model_cnt):
        with open(os.path.join(models_dir, f"verified_{i}"), "rb") as f:
            verification_summary = pickle.load(f)["summary"]
        labels = np.load(os.path.join(models_dir, f"{i}_predictions.npy"))
        verified_indices = []
        for x in verification_summary:
            if "safe" in x and "unsafe" not in x:
                verified_indices.extend(verification_summary[x])
        verified_indices = set(verified_indices)
        print(dict((x, len(v)) for x, v in verification_summary.items()))
        certified_correct = 0
        certified_wrong = 0
        for idx in range(len(res)):
            if idx in verified_indices:
                noise_res[idx, labels[idx]] += 1
                if labels[idx] == res[idx, -1]:
                    certified_correct += 1
                else:
                    certified_wrong += 1
            else:
                noise_res[idx, -2] += 1
        print(f"Approximate certified correct: {(certified_correct - certified_wrong / 9) / len(res) * 100}")
    print(noise_res[:10])
    np.save(os.path.join(exp_dir, "aggre_res.npy"), (res, noise_res))
