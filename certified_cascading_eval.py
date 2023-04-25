import argparse
import os
import json
import numpy as np

from certified_eval import precompute_DPA

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_dirs", type=str, help="dir for loading the aggregate results and commandline args",
                        required=True)
    parser.add_argument("--save_dir", type=str, help="dir for saving results", required=True)
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
    parser.add_argument("--eval_class_only", default=None, type=int,
                        help="evaluate on this class only, None for all the classes. Useful in computing FP, TP, FN, TN "
                             "in binary classification"
                        )
    parser.add_argument("--eval_partition", default=None, type=str,
                        help="evaluate on this partition only, None for all. Will be calculated before the argument eval_class_only")

    args = parser.parse_args()
    args.load_dirs = eval(args.load_dirs)
    cache_filename = os.path.join(args.save_dir, args.cache_filename)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
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
    cascading_res = None
    for load_dir in args.load_dirs:
        with open(os.path.join(load_dir, "commandline_args.txt"), 'r') as f:
            conf = args.confidence  # override the confidence
            args.__dict__.update(json.load(f))
            args.confidence = conf
        if args.parallel_precompute is not None:
            assert args.parallel_precompute_id is not None and 0 <= args.parallel_precompute_id < args.parallel_precompute
        poisoned_ins_num_range = range(args.poisoned_ins_num_st, args.poisoned_ins_num_en + 1,
                                       args.poisoned_ins_num_step)

        if not args.eval_noise:
            res, _ = np.load(os.path.join(load_dir, "aggre_res.npy"))
        else:
            _, res = np.load(os.path.join(load_dir, "aggre_res.npy"))
        if args.eval_partition is not None:
            res = res[eval(args.eval_partition)]
        if args.eval_class_only is not None:
            res = res[res[:, -1] == args.eval_class_only]

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
        elif args.dataset == "cifar10":
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

        if args.select_strategy == "DPA":
            if not args.draw_only:
                single_res = np.array(precompute_DPA(res, False))
            else:
                raise NotImplementedError
            if cascading_res is None:
                cascading_res = single_res
            else:
                print(cascading_res[:10], single_res[:10])
                is_abs_max = np.abs(cascading_res + 1.5) > np.abs(single_res + 1.5)
                cascading_res = cascading_res * is_abs_max + single_res * (1 - is_abs_max)
                print(cascading_res[:10])

        else:
            raise NotImplementedError

    cascading_res = sorted(list(cascading_res))
    np.save(cache_filename, cascading_res)
