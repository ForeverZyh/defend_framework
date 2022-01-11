import argparse
import json
import os
from fractions import Fraction

import numpy as np
from scipy.stats import beta

from cal_bound import BoundCalculator


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
        # p + p' <= 1 if n_classes == 2, else p + p' == 1
        # p - p' >= ?
        # E.g., p_a = 0.7, p_b = 0.4, p - p' >= 0.7 - 0.3 = 0.4
        # E.g., p_a = 0.7, p_b = 0.2, p - p' >= 0.7 - 0.2 = 0.5
        if n_classes > 2:
            delta_lower_bound = p_a - min(p_b, 1 - p_a) - 2e-50
        else:
            delta_lower_bound = max(p_a, 1 - p_b) - min(p_b, 1 - p_a) - 2e-50

        if delta_lower_bound > delta:  # not abstain
            if majority == res[i][-1]:
                ret[i] = 1
            else:
                ret[i] = -1
        else:
            ret[i] = 0

    return ret


def get_abstain_bagging_replace_feature_flip(res, conf, poisoned_ins_num, poisoned_feat_num, bound_cal):
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
        p_a = max(p_a, 1 - p_b)
        if p_a >= bound_cal.get_pa_lb(poisoned_ins_num, poisoned_feat_num):
            if majority == res[i][-1]:
                ret[i] = 1
            else:
                ret[i] = -1
        else:
            ret[i] = 0

    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_dir", default=None, type=str,
                        help="dir for loading the aggregate results and commandline args")
    parser.add_argument("--confidence", default=0.999, type=float,
                        help="confidence level = 1 - alpha, where alpha is the significance"
                        )
    parser.add_argument("--poisoned_feat_num", default=None, type=int,
                        help="the poisoned feature number. None means all poisoned"
                        )
    parser.add_argument("--poisoned_ins_num_st", default=0, type=int,
                        help="the start range of the poisoned instance number."
                        )
    parser.add_argument("--poisoned_ins_num_en", default=200, type=int,
                        help="the end range of the poisoned instance number. (inclusive)"
                        )
    parser.add_argument("--poisoned_ins_num_step", default=1, type=int,
                        help="the step of the poisoned instance number."
                        )

    args = parser.parse_args()
    with open(os.path.join(args.load_dir, "commandline_args.txt"), 'r') as f:
        conf = args.confidence  # override the confidence
        args.__dict__.update(json.load(f))
        args.confidence = conf
    poisoned_ins_num_range = range(args.poisoned_ins_num_st, args.poisoned_ins_num_en + 1, args.poisoned_ins_num_step)
    cache_file_name = os.path.join(args.load_dir, f"plot_{args.poisoned_feat_num}")
    if os.path.exists(cache_file_name + ".npy"):
        cache = np.load(cache_file_name + ".npy", allow_pickle=True).item()
    else:
        cache = dict()
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


    if args.select_strategy == "bagging_replace" and (args.noise_strategy is None or args.poisoned_feat_num is None):
        for poison_ins_num in poisoned_ins_num_range:
            if poison_ins_num in cache:
                ret = cache[poison_ins_num]
            else:
                ret = get_abstain_bagging_replace(res, args.confidence, args.k, poison_ins_num, args.D)
                cache[poison_ins_num] = ret
                np.save(cache_file_name, cache)

            output(ret)
    elif args.select_strategy == "bagging_replace" and args.noise_strategy == "feature_flipping":
        Ia = Fraction(int(args.alpha * 100), 100)
        bound_cal = BoundCalculator(Ia, (1 - Ia) / args.K, args.dataset, args.D, args.d, args.K, args.k,
                                    considered_degree=2, algorithm="NP+KL")
        for poison_ins_num in poisoned_ins_num_range:
            if poison_ins_num in cache:
                ret = cache[poison_ins_num]
            else:
                ret = get_abstain_bagging_replace_feature_flip(res, args.confidence, poison_ins_num,
                                                               args.poisoned_feat_num, bound_cal)
                cache[poison_ins_num] = ret
                np.save(cache_file_name, cache)

            output(ret)
    else:
        raise NotImplementedError
