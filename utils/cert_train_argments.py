import argparse


def get_arguments():
    parser = argparse.ArgumentParser()

    # general training parameters
    parser.add_argument("-d", "--dataset",
                        choices=["ember", "mnist", "mnist17", "mnist01", "imdb", "ember_limited", "mnist17_limited"],
                        help="dataset type", required=True)
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--epochs", type=int, default=200, help="training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu id for training")
    parser.add_argument("--data_aug", action='store_true', help="whether to use data augmentation")
    parser.add_argument("--no_eval_noise", action='store_true', help="whether not to evaluate on noise test data")

    # poisoning defence parameters
    parser.add_argument('--k', action='store', default=None, type=int,
                        help='number of (expected) examples in a bag')
    parser.add_argument("--select_strategy", default=None,
                        choices=["bagging_replace", "bagging_wo_replace", "binomial"],
                        help="selection strategy")
    parser.add_argument("--noise_strategy", default=None,
                        choices=["feature_flipping", "label_flipping", "all_flipping", "RAB_gaussian", "RAB_uniform"
                            , "sentence_select"],
                        help="noise strategy")
    parser.add_argument('--K', action='store', default=2, type=int,
                        help='number of bins for discretization')
    parser.add_argument('--alpha', action='store', default=0.8, type=float,
                        help='probability of the feature remains its original value')
    parser.add_argument('--sigma', action='store', default=1, type=float,
                        help='sigma for Gaussian noise')
    parser.add_argument('--a', action='store', default=0, type=float,
                        help='low for uniform noise')
    parser.add_argument('--b', action='store', default=1, type=float,
                        help='high for uniform noise')
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

    return parser
