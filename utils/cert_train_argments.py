import argparse


def get_arguments():
    parser = argparse.ArgumentParser()

    # general training parameters
    parser.add_argument("-d", "--dataset",
                        choices=["ember", "mnist", "mnist17", "mnist01", "imdb", "ember_limited", "mnist17_limited",
                                 "fmnist", "cifar10", "contagio", "cifar10-02"],
                        help="dataset type", required=True)
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--epochs", type=int, default=200, help="training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu id for training")
    parser.add_argument("--data_aug", action='store_true', help="whether to use data augmentation")
    parser.add_argument("--no_eval_noise", action='store_true', help="whether not to evaluate on noise test data")
    parser.add_argument("--fix_noise", action='store_true', help="whether to fix noise when testing")

    # poisoning defence parameters
    parser.add_argument("--select_strategy", default=None,
                        choices=["bagging_replace", "bagging_wo_replace", "binomial", "DPA", "FPA"],
                        help="selection strategy")
    parser.add_argument("--noise_strategy", default=None,
                        choices=["feature_flipping", "label_flipping", "all_flipping", "RAB_gaussian", "RAB_uniform"
                            , "sentence_select"],
                        help="noise strategy")
    # bagging
    parser.add_argument('--k', action='store', default=None, type=int,
                        help='number of (expected) examples in a bag')
    # Flipping
    parser.add_argument('--K', action='store', default=2, type=int,
                        help='number of bins for discretization')
    parser.add_argument('--alpha', action='store', default=0.8, type=float,
                        help='probability of the feature remains its original value')
    parser.add_argument('--test_alpha', action='store', default=None, type=float,
                        help='if set, use this alpha for test time noise')
    # Gaussian
    parser.add_argument('--sigma', action='store', default=1, type=float,
                        help='sigma for Gaussian noise')
    # Uniform[a,b]
    parser.add_argument('--a', action='store', default=0, type=float,
                        help='low for uniform noise')
    parser.add_argument('--b', action='store', default=1, type=float,
                        help='high for uniform noise')
    # NLP inserted poisoning word triggers
    parser.add_argument('--l', action='store', default=100, type=int,
                        help='selected segment length from the input sentence')
    parser.add_argument('--L', action='store', default=200, type=int,
                        help='max length for the input sentence')

    # certification parameters
    parser.add_argument("--N", default=1000, type=int,
                        help="number of classifiers to train"
                        )

    # dirs and files
    parser.add_argument("--load_poison_dir", default=None, type=str,
                        help="directory containing poisoned data"
                        )
    parser.add_argument("--model_save_dir", default=None, type=str, help="dir for saving the model for attacking")
    parser.add_argument("--res_save_dir", default=None, type=str, help="dir for saving the aggregate results")
    parser.add_argument("--exp_name", default=None, type=str, help="name for this experiment")
    parser.add_argument("--ember_data_dir", default="/tmp", type=str, help="dir to store cached ember dataset")
    parser.add_argument("--contagio_data_dir", default="/tmp", type=str, help="dir to store cached contagio dataset")

    # auto_LiRPA arguments
    parser.add_argument("--no_lirpa", action='store_true', help="no cert for evasion attack. only used in DPA training")
    parser.add_argument("--model", type=str, default='mlp_3layer', help='model for training')
    parser.add_argument("--eps", type=float, default=0.3, help='Target training epsilon')
    parser.add_argument("--eps_train_ratio", type=float, default=1, help='The ratio to enlarge the eps for training')
    parser.add_argument("--norm", type=float, default='inf', help='p norm for epsilon perturbation')
    parser.add_argument("--bound_type", type=str, default="CROWN-IBP",
                        choices=["IBP", "CROWN-IBP", "CROWN", "CROWN-FAST"], help='method of bound analysis')
    parser.add_argument("--scheduler_name", type=str, default="SmoothedScheduler",
                        choices=["LinearScheduler", "AdaptiveScheduler", "SmoothedScheduler", "FixedScheduler"],
                        help='epsilon scheduler')
    parser.add_argument("--scheduler_opts", type=str, default="start=3,length=60",
                        help='options for epsilon scheduler')
    parser.add_argument("--bound_opts", type=str, default=None, choices=["same-slope", "zero-lb", "one-lb"],
                        help='bound options')
    parser.add_argument("--conv_mode", type=str, choices=["matrix", "patches"], default="patches")
    parser.add_argument("--SABR", action='store_true', help="do SABR training")
    parser.add_argument("--SABR_lambda", type=float, default=0.4, help='lambda for SABR')
    parser.add_argument("--SABR_alpha", type=float, default=0.5, help='alpha for SABR')
    parser.add_argument("--SABR_step", type=int, default=8, help='step for SABR')

    return parser
