import argparse
import json
import os
from fractions import Fraction

import numpy as np
from scipy.stats import beta
from tqdm import tqdm

from utils.cal_bound import FlipBoundCalculator, SelectBoundCalculator, BoundCalculator
from utils import FEATURE_DATASET


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
            p_a, p_b = get_pa_pb(top_1, top_2, bags, alpha, n_classes)
            delta_lower_bound = p_a - p_b - 2e-50

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
            p_a, p_b = get_pa_pb(top_1, top_2, bags, alpha, n_classes)
            if majority == res[i][-1]:
                ret[i] = 1
            else:
                ret[i] = -1
            ori = ret[i]
            if (poisoned_feat_num, top_1, top_2, bags) in bound_cal.stats_cache:
                if bound_cal.stats_cache[(poisoned_feat_num, top_1, top_2, bags)] < poisoned_ins_num:
                    ret[i] = 0
            else:
                if n_classes == 2:
                    if p_a < bound_cal.get_pa_lb_binary(poisoned_ins_num):  # abstain
                        ret[i] = 0
                else:
                    if not bound_cal.check_radius(poisoned_ins_num, p_a, p_b):  # abstain
                        ret[i] = 0
            metric.update(ori, ret[i])
            progress_bar.set_postfix(metric.get_postfix())
            progress_bar.update(1)

    return ret


def get_pa_pb(top_1, top_2, bags, alpha, n_classes):
    if top_1 == bags:
        p_a = np.power(alpha / n_classes, 1.0 / bags)
        p_b = 1 - p_a
    else:
        p_a = beta.ppf(alpha / n_classes, top_1, bags - top_1 + 1)  # p >= p_a
        p_b = beta.ppf(1 - alpha / n_classes, top_2 + 1, bags - top_2)  # p' <= p_b
    # p + p' <= 1 if n_classes == 2, else p + p' == 1
    if n_classes == 2:
        p_a = max(p_a, 1 - p_b)
    p_b = min(p_b, 1 - p_a)
    return p_a, p_b


def precompute_bag(res, conf, ex_in_bag, D):
    # res.shape: (n_examples, n_classes + 1)
    ret = np.ones(res.shape[0])
    alpha = (1 - conf) / res.shape[0]
    n_classes = res.shape[1] - 1
    bags = np.sum(res[0][:-1])
    stats_cache = {}
    for i in range(len(res)):
        majority = np.argmax(res[i, :-1])
        top_1 = res[i][majority]
        top_2 = max(res[i][j] for j in range(n_classes) if j != majority)
        p_a, p_b = get_pa_pb(top_1, top_2, bags, alpha, n_classes)
        if (0, top_1, top_2, bags) not in stats_cache:
            stats_cache[(0, top_1, top_2, bags)] = np.ceil(
                D * (1 - np.power(1 - (p_a - p_b) / 2, 1.0 / ex_in_bag)) - 1)

    # compute stats
    cal_statistics(res, bags, alpha, n_classes, stats_cache, 0)


def precompute_binary(res, conf, bound_cal: BoundCalculator, parallel_num=None, parallel_id=None):
    # res.shape: (n_examples, n_classes + 1)
    ret = np.ones(res.shape[0])
    alpha = (1 - conf) / res.shape[0]
    n_classes = res.shape[1] - 1
    assert n_classes == 2
    bags = np.sum(res[0][:-1])
    tops = sorted(list(set([res[i][np.argmax(res[i][:-1])] for i in range(len(res))])))
    if parallel_num is not None:
        tops = tops[parallel_id::parallel_num]
    with tqdm(total=len(tops)) as progress_bar:
        for top_1 in tops:
            top_2 = bags - top_1
            p_a, p_b = get_pa_pb(top_1, top_2, bags, alpha, n_classes)
            pre_res = bound_cal.get_poisoned_ins_binary(top_1, top_2, p_a, bags, parallel_num=parallel_num,
                                                        parallel_id=parallel_id)
            progress_bar.set_postfix({"ins": pre_res, "top_1": top_1})
            progress_bar.update(1)
    # compute stats 
    if parallel_num is not None:
        return
    cal_statistics(res, bags, alpha, n_classes, bound_cal.stats_cache, bound_cal.s)


def precompute(res, conf, bound_cal: BoundCalculator, parallel_num=None, parallel_id=None):
    # res.shape: (n_examples, n_classes + 1)
    ret = np.ones(res.shape[0])
    alpha = (1 - conf) / res.shape[0]
    n_classes = res.shape[1] - 1
    assert n_classes > 2
    bags = np.sum(res[0][:-1])
    tops = {}
    for i in range(len(res)):
        majority = np.argmax(res[i, :-1])
        top_1 = res[i][majority]
        if top_1 not in tops:
            tops[top_1] = []
        top_2 = max(res[i][j] for j in range(n_classes) if j != majority)
        tops[top_1].append(top_2)

    for top_1 in tops:
        tops[top_1].sort(key=lambda x: -x)
    tops_key = sorted(list(tops.keys()))

    if parallel_num is not None:
        tops_key = tops_key[parallel_id::parallel_num]
    with tqdm(total=sum(len(tops[top_1]) for top_1 in tops_key)) as progress_bar:
        for top_1 in tops_key:
            for top_2 in tops[top_1]:
                p_a, p_b = get_pa_pb(top_1, top_2, bags, alpha, n_classes)
                pre_res = bound_cal.get_poisoned_ins(top_1, top_2, p_a, p_b, bags, parallel_num=parallel_num,
                                                     parallel_id=parallel_id)
                progress_bar.set_postfix({"ins": pre_res, "top_1": top_1, "top_2": top_2})
                progress_bar.update(1)
    # compute stats 
    if parallel_num is not None:
        return
    cal_statistics(res, bags, alpha, n_classes, bound_cal.stats_cache, bound_cal.s)


def cal_statistics(res, bags, alpha, n_classes, cache, s):
    cor_cnt = 0
    auc = 0
    radius = []
    for i in range(len(res)):
        majority = np.argmax(res[i, :-1])
        top_1 = res[i][majority]
        top_2 = max(res[i][j] for j in range(n_classes) if j != majority)
        p_a, p_b = get_pa_pb(top_1, top_2, bags, alpha, n_classes)
        if majority == res[i, -1]:
            if p_a > p_b:
                cor_cnt += 1
                auc += cache[(s, top_1, top_2, bags)]
                radius.append(cache[(s, top_1, top_2, bags)])
            else:
                radius.append(0)
        else:
            radius.append(0)

    radius.sort()
    mcr = (radius[len(res) // 2 - 1] + radius[len(res) // 2]) / 2.0 if len(res) % 2 == 0 else radius[len(res) // 2]
    print(f"Normal Acc: {cor_cnt * 100.0 / len(res):.2f}\tAUC: {auc * 1.0 / len(res):.2f}\tMCR: {mcr:.1f}")


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
    parser.add_argument("--k_upperbound", default=np.inf, type=int,
                        help="the upper bound of k used to control the computation efficiency."
                        )
    parser.add_argument("--parallel_precompute", default=None, type=int,
                        help="the number of manual split for the parallel. None for not parallel"
                        )
    parser.add_argument("--parallel_precompute_id", default=None, type=int,
                        help="the manual split id"
                        )
    parser.add_argument("--precompute_only", action='store_true', help="only to compute the stats not drawing")
    parser.add_argument("--draw_only", action='store_true', help="only to draw not precomputing the stats")
    parser.add_argument("--eval_noise", action='store_true', help="evaluate on noise prediction (backdoor attacks)")

    args = parser.parse_args()
    with open(os.path.join(args.load_dir, "commandline_args.txt"), 'r') as f:
        conf = args.confidence  # override the confidence
        args.__dict__.update(json.load(f))
        args.confidence = conf
    if args.parallel_precompute is not None:
        assert args.parallel_precompute_id is not None and 0 <= args.parallel_precompute_id < args.parallel_precompute
    poisoned_ins_num_range = range(args.poisoned_ins_num_st, args.poisoned_ins_num_en + 1, args.poisoned_ins_num_step)
    cache_filename = os.path.join(args.load_dir, args.cache_filename)
    if os.path.exists(cache_filename + ".npy") and not args.precompute_only:
        respond = input("Experiment already exists, type [O] to overwrite, type [R] to resume")
        if respond == "O":
            cache = dict()
        elif respond == "R":
            cache = np.load(cache_filename + ".npy", allow_pickle=True).item()
        else:
            exit(0)
    else:
        cache = dict()
    if not args.eval_noise:
        res, _ = np.load(os.path.join(args.load_dir, "aggre_res.npy"))
    else:
        _, res = np.load(os.path.join(args.load_dir, "aggre_res.npy"))

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
    elif args.dataset == "mnist17_limited":
        args.D = 100
        if args.noise_strategy is not None:
            if args.noise_strategy == "feature_flipping":
                args.d = 28 * 28
            elif args.noise_strategy == "all_flipping":
                args.d = 28 * 28 + 1
            elif args.noise_strategy == "label_flipping":
                args.d = 1
            else:
                raise NotImplementedError
    elif args.dataset == "mnist01":
        args.D = 12665
        if args.noise_strategy is not None:
            if args.noise_strategy == "feature_flipping":
                args.d = 28 * 28
            elif args.noise_strategy == "all_flipping":
                args.d = 28 * 28 + 1
            elif args.noise_strategy == "label_flipping":
                args.d = 1
            else:
                raise NotImplementedError
    elif args.dataset in ["mnist", "fmnist"]:
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
    elif args.dataset == "ember_limited":
        args.D = 600000
        if args.noise_strategy is not None:
            if args.noise_strategy == "feature_flipping":
                args.d = 17
            elif args.noise_strategy == "all_flipping":
                args.d = 17 + 1
            elif args.noise_strategy == "label_flipping":
                args.d = 1
            else:
                raise NotImplementedError
    else:
        raise NotImplementedError

    if args.select_strategy == "bagging_replace" and (args.noise_strategy is None or args.poisoned_feat_num is None):
        if not args.draw_only:
            precompute_bag(res, args.confidence, args.k, args.D)
        if args.precompute_only:
            exit(0)
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
        if args.dataset in FEATURE_DATASET:
            Ia = Fraction(int(args.alpha * 100), 100)
            bound_cal = FlipBoundCalculator(Ia, (1 - Ia) / args.K, args.dataset, args.D, args.d, args.K, args.k,
                                            args.poisoned_feat_num, args.eval_noise, args.k_upperbound)
        elif args.dataset == "imdb":
            if args.noise_strategy == "sentence_select":
                bound_cal = SelectBoundCalculator(None, args.dataset, args.D, args.L, args.k, args.l,
                                                  args.poisoned_feat_num)
            else:
                Ia = Fraction(int(args.alpha * 100), 100)
                bound_cal = SelectBoundCalculator((Ia, (1 - Ia) / args.K, args.K), args.dataset, args.D, args.L, args.k,
                                                  args.l, args.poisoned_feat_num)
        else:
            raise NotImplementedError

        if not args.draw_only:
            if res.shape[1] - 1 == 2:  # n_classes == 2
                precompute_binary(res, args.confidence, bound_cal, parallel_num=args.parallel_precompute,
                                  parallel_id=args.parallel_precompute_id)
            else:
                precompute(res, args.confidence, bound_cal, parallel_num=args.parallel_precompute,
                           parallel_id=args.parallel_precompute_id)
        if args.precompute_only:
            exit(0)
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
