import argparse
import json
import os
from fractions import Fraction

import numpy as np
from scipy.stats import beta


def get_abstain_bagging_replace(res, conf, ex_in_bag, poison_ins_num, D, poison_feat_num=None, d=None):
    delta = (1 - ((1 - Fraction(poison_ins_num, D)) ** ex_in_bag)) * 2
    # res.shape: (n_examples, n_classes + 1)
    ret = np.ones(res.shape[0])
    alpha = (1 - conf) / res.shape[0]
    n_classes = res.shape[1] - 1
    bags = np.sum(res[0][:-1])
    for i in range(len(res)):
        majority = np.argmax(res[i][:-1])
        top_1 = res[i][majority]
        top_2 = max(res[i][j] for j in range(n_classes) if j != majority)
        p_a = beta.ppf(alpha / n_classes, top_1, bags - top_1 + 1)  # p \in [p_a, 1]
        if top_2 > 0:
            p_b = beta.ppf(1 - alpha / n_classes, top_2, bags - top_2 + 1)  # p' \in [0, p_b]
        else:
            p_b = 1
        # p + p' <= 1
        # p - p' >= ?
        # E.g., p_a = 0.7, p_b = 0.4, p - p' >= 0.7 - 0.3 = 0.4
        # E.g., p_a = 0.7, p_b = 0.2, p - p' >= 0.8 - 0.2 = 0.6
        delta_lower_bound = max(p_a, 1 - p_b) - min(p_b, 1 - p_a)

        if delta_lower_bound > delta:  # not abstain
            if majority == res[i][-1]:
                ret[i] = 1
            else:
                ret[i] = -1
        else:
            ret[i] = 0

    return ret


def get_abstain_bagging_replace_feature_flip(res, conf, ex_in_bag, poison_ins_num, D, poison_feat_num, d):
    # res.shape: (n_examples, n_classes + 1)
    ret = np.ones(res.shape[0])
    alpha = (1 - conf) / res.shape[0]
    n_classes = res.shape[1] - 1
    bags = np.sum(res[0][:-1])
    for i in range(len(res)):
        majority = np.argmax(res[i][:-1])
        top_1 = res[i][majority]
        top_2 = max(res[i][j] for j in range(n_classes) if j != majority)
        p_a = beta.ppf(alpha / n_classes, top_1, bags - top_1 + 1)
        if top_2 > 0:
            p_b = beta.ppf(1 - alpha / n_classes, top_2, bags - top_2 + 1)  # p' \in [0, p_b]
        else:
            p_b = 1
        p_a = max(p_a, 1 - p_b)
        ret[i] = p_a >= Fraction(209113655259859, 281474976710656)

    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_dir", default=None, type=str,
                        help="dir for loading the aggregate results and commandline args")
    parser.add_argument("--confidence", default=0.999, type=float,
                        help="confidence level = 1 - alpha, where alpha is the significance"
                        )

    args = parser.parse_args()
    with open(os.path.join(args.load_dir, "commandline_args.txt"), 'r') as f:
        conf = args.confidence  # override the confidence
        load_dir = args.load_dir
        args.__dict__ = json.load(f)
        args.confidence = conf
        args.load_dir = load_dir

    res = np.load(os.path.join(args.load_dir, "aggre_res.npy"))
    if args.dataset == "mnist17":
        args.D = 13007
        args.d = 28 * 28 + 1
    else:
        raise NotImplementedError


    def output(x):
        print(f"Certified Accuracy: {np.mean(x == 1) * 100:.2f}%, "
              f"Abstained: {np.mean(x == 0) * 100:.2f}%, "
              f"Wrong: {np.mean(x == -1) * 100:.2f}%")


    if args.select_strategy == "bagging_replace" and args.noise_strategy is None:
        for poison_ins_num in range(0, 150, 10):
            ret = get_abstain_bagging_replace(res, args.confidence, args.k, poison_ins_num, args.D)
            output(ret)
    elif args.select_strategy == "bagging_replace" and args.noise_strategy == "feature_flipping":
        pass
    else:
        raise NotImplementedError
