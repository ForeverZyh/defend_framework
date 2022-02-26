import argparse
import json
import os
from fractions import Fraction

import numpy as np
from scipy.stats import beta
from tqdm import tqdm

from cal_bound import FlipBoundCalculator, SelectBoundCalculator, BoundCalculator


def output(x):
    print(f"Certified Accuracy: {np.mean(x == 1) * 100:.2f}%, "
          f"Abstained: {np.mean(x == 0) * 100:.2f}%, "
          f"Wrong: {np.mean(x == -1) * 100:.2f}%")


class Metric:
    def __init__(self):
        self.ori_acc_cnt = 0
        self.cert_acc_cnt = 0
        self.abstain_cnt = 0
        self.wrong_cnt = 0
        self.cnt = 0

    def update(self, ori, final):
        self.cnt += 1
        if ori == 1:
            self.ori_acc_cnt += 1
        if final == 0:
            self.abstain_cnt += 1
        elif final == 1:
            self.cert_acc_cnt += 1
        else:
            self.wrong_cnt += 1

    def get_postfix(self):
        return {'cert_acc': self.cert_acc_cnt / self.cnt, 'abstain_r': self.abstain_cnt / self.cnt,
                'wrong_r': self.wrong_cnt / self.cnt, 'ori_acc': self.ori_acc_cnt / self.cnt}


def get_abstain_bagging_replace(res, conf, ex_in_bag, poison_ins_num, D, poison_feat_num=None, d=None):
    delta = (1 - ((1 - Fraction(poison_ins_num, D)) ** ex_in_bag)) * 2
    # res.shape: (n_examples, n_classes + 1)
    ret = np.ones(res.shape[0])
    alpha = (1 - conf) / res.shape[0]
    n_classes = res.shape[1] - 1
    bags = np.sum(res[0][:-1])

    with tqdm(total=len(res)) as progress_bar:
        metric = Metric()
        for i in range(len(res)):
            majority = np.argmax(res[i][:-1])
            top_1 = res[i][majority]
            top_2 = max(res[i][j] for j in range(n_classes) if j != majority)
            if top_1 == bags:
                p_a = np.power(alpha / n_classes, 1.0 / bags)
                p_b = 1 - p_a
            else:
                p_a = beta.ppf(alpha / n_classes, top_1, bags - top_1 + 1)  # p >= p_a
                p_b = beta.ppf(1 - alpha / n_classes, top_2 + 1, bags - top_2)  # p' <= p_b
            # p + p' <= 1 if n_classes == 2, else p + p' == 1
            # p - p' >= ?
            # E.g., p_a = 0.7, p_b = 0.4, p - p' >= 0.7 - 0.3 = 0.4
            # E.g., p_a = 0.7, p_b = 0.2, p - p' >= 0.7 - 0.2 = 0.5
            if n_classes > 2:
                delta_lower_bound = p_a - min(p_b, 1 - p_a) - 2e-50
            else:
                delta_lower_bound = max(p_a, 1 - p_b) - min(p_b, 1 - p_a) - 2e-50

            if majority == res[i][-1]:
                ret[i] = 1
            else:
                ret[i] = -1
            ori = ret[i]
            if delta_lower_bound <= delta:  # abstain
                ret[i] = 0
            metric.update(ori, ret[i])
            progress_bar.set_postfix(metric.get_postfix())
            progress_bar.update(1)

    return ret


def get_abstain_bagging_replace_feature_flip(res, conf, poisoned_ins_num, poisoned_feat_num,
                                             bound_cal: BoundCalculator):
    # res.shape: (n_examples, n_classes + 1)
    ret = np.ones(res.shape[0])
    alpha = (1 - conf) / res.shape[0]
    n_classes = res.shape[1] - 1
    bags = np.sum(res[0][:-1])
    with tqdm(total=len(res)) as progress_bar:
        metric = Metric()
        for i in range(len(res)):
            majority = np.argmax(res[i][:-1])
            top_1 = res[i][majority]
            top_2 = max(res[i][j] for j in range(n_classes) if j != majority)
            if top_1 == bags:
                p_a = np.power(alpha / n_classes, 1.0 / bags)
                p_b = 1 - p_a
            else:
                p_a = beta.ppf(alpha / n_classes, top_1, bags - top_1 + 1)  # p >= p_a
                p_b = beta.ppf(1 - alpha / n_classes, top_2 + 1, bags - top_2)  # p' <= p_b
            if n_classes == 2:
                p_a = max(p_a, 1 - p_b)
            p_b = min(p_b, 1 - p_a)

            if majority == res[i][-1]:
                ret[i] = 1
            else:
                ret[i] = -1
            ori = ret[i]
            if n_classes == 2:
                if p_a < bound_cal.get_pa_lb_binary(poisoned_ins_num, poisoned_feat_num):  # abstain
                    ret[i] = 0
            else:
                if not bound_cal.check_radius(poisoned_ins_num, poisoned_feat_num, p_a, p_b):  # abstain
                    ret[i] = 0
            metric.update(ori, ret[i])
            progress_bar.set_postfix(metric.get_postfix())
            progress_bar.update(1)

    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_dir", default=None, type=str,
                        help="dir for loading the aggregate results and commandline args")
    parser.add_argument("--cache_filename", default=None, type=str,
                        help="name to cache the file", required=True)
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
    cache_filename = os.path.join(args.load_dir, args.cache_filename)
    if os.path.exists(cache_filename + ".npy"):
        respond = input("Experiment already exists, type [O] to overwrite, type [R] to resume")
        if respond == "O":
            cache = dict()
        elif respond == "R":
            cache = np.load(cache_filename + ".npy", allow_pickle=True).item()
        else:
            exit(0)
    else:
        cache = dict()
    res, res_noise = np.load(os.path.join(args.load_dir, "aggre_res.npy"))
    if args.dataset == "mnist17":
        args.D = 13007
        if args.noise_strategy is not None:
            if args.noise_strategy == "feature_flipping":
                args.d = 28 * 28
            elif args.noise_strategy == "all_flipping":
                args.d = 28 * 28 + 1
            elif args.noise_strategy == "label_flipping":
                args.d = 1
            else:
                raise NotImplementedError

    elif args.dataset == "mnist":
        args.D = 60000
        if args.noise_strategy is not None:
            if args.noise_strategy == "feature_flipping":
                args.d = 28 * 28
            elif args.noise_strategy == "all_flipping":
                args.d = 28 * 28 + 1
            elif args.noise_strategy == "label_flipping":
                args.d = 1
            else:
                raise NotImplementedError
    elif args.dataset == "imdb":
        args.D = 25000
        args.d = None
    elif args.dataset == "ember":
        args.D = 600000
        if args.noise_strategy is not None:
            if args.noise_strategy == "feature_flipping":
                args.d = 2351
            elif args.noise_strategy == "all_flipping":
                args.d = 2351 + 1
            elif args.noise_strategy == "label_flipping":
                args.d = 1
            else:
                raise NotImplementedError
    else:
        raise NotImplementedError

    if args.select_strategy == "bagging_replace" and (args.noise_strategy is None or args.poisoned_feat_num is None):
        for poison_ins_num in poisoned_ins_num_range:
            if poison_ins_num in cache:
                ret = cache[poison_ins_num]
            else:
                ret = get_abstain_bagging_replace(res, args.confidence, args.k, poison_ins_num, args.D)
                cache[poison_ins_num] = ret
                np.save(cache_filename, cache)

            # output(ret)
    elif args.select_strategy == "bagging_replace" and args.noise_strategy in ["feature_flipping", "label_flipping",
                                                                               "all_flipping", "sentence_select"]:
        if args.dataset in ["mnist", "mnist17", "ember"]:
            Ia = Fraction(int(args.alpha * 100), 100)
            bound_cal = FlipBoundCalculator(Ia, (1 - Ia) / args.K, args.dataset, args.D, args.d, args.K, args.k)
        elif args.dataset == "imdb":
            if args.noise_strategy == "sentence_select":
                bound_cal = SelectBoundCalculator(None, args.dataset, args.D, args.L, args.k, args.l)
            else:
                Ia = Fraction(int(args.alpha * 100), 100)
                bound_cal = SelectBoundCalculator((Ia, (1 - Ia) / args.K, args.K), args.dataset, args.D, args.L, args.k,
                                                  args.l)

        for poison_ins_num in poisoned_ins_num_range:
            if poison_ins_num in cache:
                ret = cache[poison_ins_num]
            else:
                ret = get_abstain_bagging_replace_feature_flip(res, args.confidence, poison_ins_num,
                                                               args.poisoned_feat_num, bound_cal)
                cache[poison_ins_num] = ret
                np.save(cache_filename, cache)

            # output(ret)
    else:
        raise NotImplementedError
