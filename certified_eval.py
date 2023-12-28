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


def get_abstain_DPA(res, poison_ins_num):
    # res.shape: (n_examples, n_classes + 2)
    ret = np.ones(res.shape[0])

    with tqdm(total=len(res)) as progress_bar:
        metric = Metric()
        for i in range(len(res)):
            majority = np.argmax(res[i][:-2])
            top_1 = res[i][majority]
            runner_up = min([j for j in range(res.shape[1] - 2) if j != majority])
            top_2 = res[i][runner_up]
            N_3 = res[i][-2]

            if majority == res[i][-1]:
                ret[i] = 1
            else:
                ret[i] = -1
            ori = ret[i]
            r = top_1 - top_2 - N_3 - (majority > runner_up)
            if r < 0 or r // 2 // 2 < poison_ins_num:
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


def precompute_DPA(res, in_place, at=None, sort=True, ret_dict=None, print_out=True):
    # res.shape: (n_examples, n_classes + 2)
    radius = []
    cor_cnt = 0
    auc = 0

    for i in range(len(res)):
        majority = np.argmax(res[i][:-2])
        runner_up = min([j for j in range(res.shape[1] - 2) if j != majority])
        top_1 = res[i][majority]
        top_2 = res[i][runner_up]
        N_3 = res[i][-2]

        if majority == res[i][-1]:
            cor_cnt += 1
            r = top_1 - top_2 - N_3 - (majority > runner_up)
            if r < 0:
                r = -1  # correct but cannot certify anything
            else:
                if in_place:
                    r = r // 2
                else:
                    r = r // 2 // 2
            radius.append(r)
            if r > 0:
                auc += r
        else:
            r = top_1 - top_2 - N_3 - (majority > runner_up)
            if r < 0:
                r = -2  # wrong, but cannot certify anything
            else:
                # wrong, and can certify r, to see the real radius, use -r - 3
                if in_place:
                    r = -3 - r // 2
                else:
                    r = -3 - r // 2 // 2
            radius.append(r)

    if sort:
        radius.sort()
        mcr = (radius[len(res) // 2 - 1] + radius[len(res) // 2]) / 2.0 if len(res) % 2 == 0 else radius[len(res) // 2]
        if print_out:
            print(f"Normal Acc: {cor_cnt * 100.0 / len(res):.2f}\tAUC: {auc * 1.0 / len(res):.2f}\tMCR: {mcr:.1f}")
        if at is not None:
            cert_acc = np.sum(np.array(radius) >= at) * 100.0 / len(res)
            cert_wrong = np.sum(-np.array(radius) - 3 >= at) * 100.0 / len(res)
            abstain = 100 - cert_acc - cert_wrong
            if ret_dict is not None:
                ret_dict["res"] = (cert_acc, cert_wrong, abstain)
            if print_out:
                print(f"Cert Acc@{at}: {round(cert_acc, 2)}")
                print(f"Cert Wrong@{at}: {round(cert_wrong, 2)}")
                print(f"Abstain@{at}: {round(abstain, 2)}")
    return radius


def precompute_DPA_tau(res, pred_and_conf, args, control_res=None, sort=True):
    n_classes = res.shape[1] - 2
    ret = {}
    if control_res is not None:
        filters = np.argmax(control_res[:, :-1], axis=1) == control_res[:, -1]
        print(np.mean(filters))
    else:
        filters = None
    for tau in np.linspace(0, 1, 101):
        res[:, :-1] = 0  # remove the prediction cnts
        for i in range(args.N):
            verified = pred_and_conf[i][1] <= tau
            pred_cert = (pred_and_conf[i][0] * verified + (1 - verified) * n_classes).astype(np.int32)
            res[np.arange(pred_cert.shape[0]), pred_cert] += 1
        # print("tau: %.2f\n" % tau)
        # print(res[:10])
        aux_dict = {}
        aux_dict1 = {}
        precompute_DPA(res, args.in_place, at=args.poisoned_ins_num, sort=sort, ret_dict=aux_dict, print_out=False)
        if filters is not None:
            precompute_DPA(res[filters], args.in_place, at=args.poisoned_ins_num, sort=sort, ret_dict=aux_dict1,
                           print_out=False)
        else:
            aux_dict1["res"] = aux_dict["res"]
        ret[tau] = (aux_dict["res"], aux_dict1["res"])
    return ret


def precompute_FPA(res, control_res=None, at=None, sort=True, print_out=True):
    # res.shape: (n_examples, n_classes + 2)
    radius = []
    cor_cnt = 0
    auc = 0

    for i in range(len(res)):
        majority = np.argmax(res[i][:-1])
        runner_up = min([j for j in range(res.shape[1] - 1) if j != majority])
        top_1 = res[i][majority]
        top_2 = res[i][runner_up]

        if majority == res[i][-1]:
            cor_cnt += 1
            r = top_1 - top_2 - (majority > runner_up)
            if r < 0:
                r = -1
            else:
                r = r // 2
            radius.append(r)
            if r > 0:
                auc += r
        else:
            r = top_1 - top_2 - (majority > runner_up)
            if r < 0:
                r = -2
            else:
                r = -3 - r // 2
            radius.append(r)

    if sort:
        radius.sort()
        mcr = (radius[len(res) // 2 - 1] + radius[len(res) // 2]) / 2.0 if len(res) % 2 == 0 else radius[len(res) // 2]
        print(f"Normal Acc: {cor_cnt * 100.0 / len(res):.2f}\tAUC: {auc * 1.0 / len(res):.2f}\tMCR: {mcr:.1f}")
        if at is not None:
            cert_acc = np.sum(np.array(radius) >= at) * 100.0 / len(res)
            cert_wrong = np.sum(-np.array(radius) - 3 >= at) * 100.0 / len(res)
            abstain = 100 - cert_acc - cert_wrong
            if print_out:
                print(f"Cert Acc@{at}: {round(cert_acc, 2)}")
                print(f"Cert Wrong@{at}: {round(cert_wrong, 2)}")
                print(f"Abstain@{at}: {round(abstain, 2)}")
    return radius


def precompute_bag(res, conf, ex_in_bag, D):
    # res.shape: (n_examples, n_classes + 1)
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
    return cal_statistics(res, bags, alpha, n_classes, stats_cache, 0)


def precompute_binary(res, conf, bound_cal: BoundCalculator, parallel_num=None, parallel_id=None):
    # res.shape: (n_examples, n_classes + 1)
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
    return cal_statistics(res, bags, alpha, n_classes, bound_cal.stats_cache, bound_cal.s)


def precompute(res, conf, bound_cal: BoundCalculator, parallel_num=None, parallel_id=None):
    # res.shape: (n_examples, n_classes + 1)
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
    return cal_statistics(res, bags, alpha, n_classes, bound_cal.stats_cache, bound_cal.s)


def cal_statistics(res, bags, alpha, n_classes, cache, s, sort=True):
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
                if cache[(s, top_1, top_2, bags)] >= 0:
                    auc += cache[(s, top_1, top_2, bags)]
                    radius.append(cache[(s, top_1, top_2, bags)])
                else:
                    radius.append(-1)
            else:
                radius.append(-2)
        else:
            if cache[(s, top_1, top_2, bags)] >= 0:
                radius.append(-2 - cache[(s, top_1, top_2, bags)])
            else:
                radius.append(-2)

    if sort:
        radius.sort()
        mcr = (radius[len(res) // 2 - 1] + radius[len(res) // 2]) / 2.0 if len(res) % 2 == 0 else radius[len(res) // 2]
        print(f"Normal Acc: {cor_cnt * 100.0 / len(res):.2f}\tAUC: {auc * 1.0 / len(res):.2f}\tMCR: {mcr:.1f}")
    return radius


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_dir", default=None, type=str,
                        help="dir for loading the aggregate results and commandline args")
    parser.add_argument("--load_model_dir", default=None, type=str,
                        help="dir for loading the model and predictions")
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
    parser.add_argument("--poisoned_ins_num", default=None, type=int,
                        help="the poisoned instance number."
                        )
    parser.add_argument("--delta_non_tight", default=0, type=float,
                        help="the upper bound of probabilities that can induce incompleteness (non-tightness). "
                             "Setting to 0 means tight certification, setting to 1e-4 can greatly improve efficiency but "
                             "not seemingly decreasing certified accuracy."
                        )
    parser.add_argument("--parallel_precompute", default=None, type=int,
                        help="the number of manual split for the parallel. None for not parallel"
                        )
    parser.add_argument("--parallel_precompute_id", default=None, type=int,
                        help="the manual split id"
                        )
    parser.add_argument("--draw_only", action='store_true', help="only to draw not precomputing the stats")
    parser.add_argument("--eval_noise", action='store_true', help="evaluate on noise prediction (backdoor attacks)")
    parser.add_argument("--in_place", action='store_true',
                        help="the backdoor attack is inplace replacement, so DPA only needs to divided by 2")
    parser.add_argument("--eval_class_only", default=None, type=int,
                        help="evaluate on this class only, None for all the classes. "
                             "Useful in computing FP, TP, FN, TN in binary classification"
                        )
    parser.add_argument("--exclude_class_only", default=None, type=int,
                        help="exclude the evaluation on this class only, None for all the classes. "
                             "Useful in computing FP, TP, FN, TN in binary classification"
                        )
    parser.add_argument("--eval_partition", default=None, type=str,
                        help="evaluate on this partition only, None for all. "
                             "Will be calculated before the argument eval_class_only")
    parser.add_argument("--control_partition", default=None, type=str,
                        help="The control set of predictions")

    args = parser.parse_args()
    with open(os.path.join(args.load_dir, "commandline_args.txt"), 'r') as f:
        conf = args.confidence  # override the confidence
        args.__dict__.update(json.load(f))
        args.confidence = conf
        if "patchguard" not in args.__dict__:
            args.patchguard = False
    if args.parallel_precompute is not None:
        assert args.parallel_precompute_id is not None and 0 <= args.parallel_precompute_id < args.parallel_precompute
    assert args.eval_class_only is None or args.exclude_class_only is None
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
    if not args.eval_noise:
        res, _ = np.load(os.path.join(args.load_dir, "aggre_res.npy"))
    else:
        _, res = np.load(os.path.join(args.load_dir, "aggre_res.npy"))
    pred_and_conf = []
    if args.load_model_dir is not None:
        for i in range(args.N):
            a, b = np.load(os.path.join(args.load_model_dir, f"{i}_predictions.npy"))
            pred_and_conf.append([a, b])

    if args.control_partition is not None:
        control_res = res[eval(args.control_partition)]
    else:
        control_res = None

    if args.eval_partition is not None:
        res = res[eval(args.eval_partition)]
        for i in range(len(pred_and_conf)):
            pred_and_conf[i][0] = pred_and_conf[i][0][eval(args.eval_partition)]
            pred_and_conf[i][1] = pred_and_conf[i][1][eval(args.eval_partition)]

    if args.eval_class_only is not None:
        idx = res[:, -1] == args.eval_class_only
        res = res[idx]
        if control_res is not None:
            control_res = control_res[idx]
        for i in range(len(pred_and_conf)):
            pred_and_conf[i][0] = pred_and_conf[i][0][idx]
            pred_and_conf[i][1] = pred_and_conf[i][1][idx]
    elif args.exclude_class_only is not None:
        idx = res[:, -1] != args.exclude_class_only
        res = res[idx]
        if control_res is not None:
            control_res = control_res[idx]
        for i in range(len(pred_and_conf)):
            pred_and_conf[i][0] = pred_and_conf[i][0][idx]
            pred_and_conf[i][1] = pred_and_conf[i][1][idx]

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
    elif args.dataset in ["cifar10", "cifar10-expand"]:
        args.D = 50000
        if args.noise_strategy is not None:
            if args.noise_strategy == "feature_flipping":
                args.d = 32 * 32 * 3
            elif args.noise_strategy == "all_flipping":
                args.d = 32 * 32 * 3 + 1
            elif args.noise_strategy == "label_flipping":
                args.d = 1
            else:
                raise NotImplementedError
    elif args.dataset == "cifar10-02":
        args.D = 10000
        if args.noise_strategy is not None:
            if args.noise_strategy == "feature_flipping":
                args.d = 32 * 32
            elif args.noise_strategy == "all_flipping":
                args.d = 32 * 32 + 1
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
    elif args.dataset == "contagio":
        args.D = 6000
        if args.noise_strategy is not None:
            if args.noise_strategy == "feature_flipping":
                args.d = 135
            elif args.noise_strategy == "all_flipping":
                args.d = 135 + 1
            elif args.noise_strategy == "label_flipping":
                args.d = 1
            else:
                raise NotImplementedError
    else:
        raise NotImplementedError

    if args.select_strategy == "bagging_replace" and (args.noise_strategy is None or args.poisoned_feat_num is None):
        if not args.draw_only:
            np.save(cache_filename, precompute_bag(res, args.confidence, args.k, args.D))
        else:
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
                                            args.poisoned_feat_num, args.eval_noise, args.delta_non_tight,
                                            args.noise_strategy)
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
                np.save(cache_filename,
                        precompute_binary(res, args.confidence, bound_cal, parallel_num=args.parallel_precompute,
                                          parallel_id=args.parallel_precompute_id))
            else:
                np.save(cache_filename,
                        precompute(res, args.confidence, bound_cal, parallel_num=args.parallel_precompute,
                                   parallel_id=args.parallel_precompute_id))
        else:
            for poison_ins_num in poisoned_ins_num_range:
                if poison_ins_num in cache:
                    ret = cache[poison_ins_num]
                else:
                    ret = get_abstain_bagging_replace_feature_flip(res, args.confidence, poison_ins_num,
                                                                   args.poisoned_feat_num, bound_cal)
                    cache[poison_ins_num] = ret
                    np.save(cache_filename, cache)

            # output(ret)
    elif args.select_strategy == "DPA":
        if args.patchguard:
            if not args.draw_only:
                np.save(cache_filename, precompute_DPA_tau(res, pred_and_conf, args, control_res))
            else:
                raise NotImplementedError
        else:
            if not args.draw_only:
                np.save(cache_filename, precompute_DPA(res, args.in_place, at=args.poisoned_ins_num))
            else:
                for poison_ins_num in poisoned_ins_num_range:
                    if poison_ins_num in cache:
                        ret = cache[poison_ins_num]
                    else:
                        ret = get_abstain_DPA(res, poison_ins_num)
                        cache[poison_ins_num] = ret
                        np.save(cache_filename, cache)
    elif args.select_strategy == "FPA":
        np.save(cache_filename, precompute_FPA(res, control_res, at=args.poisoned_ins_num))
        if control_res is not None:
            filters = np.argmax(control_res[:, :-1], axis=1) == control_res[:, -1]
            print(np.mean(filters))
            precompute_FPA(res[filters], control_res, at=args.poisoned_ins_num)
    else:
        raise NotImplementedError
