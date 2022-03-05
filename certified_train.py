import argparse
import random
import os
import json
import time

import numpy as np
import tensorflow as tf

from utils.data_processing import MNIST17DataPreprocessor, MNISTDataPreprocessor, IMDBDataPreprocessor, \
    EmberDataPreProcessor, EMBER_DATASET
from models import MNIST17Model, MNISTModel, IMDBTransformerModel, EmberModel
from utils.train_utils import train_many

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # general training parameters
    parser.add_argument("-d", "--dataset", choices=["ember", "mnist", "mnist17", "imdb", "ember_limited"],
                        help="dataset type", required=True)
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--epochs", type=int, default=200, help="training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu id for training")
    parser.add_argument("--data_aug", action='store_true', help="whether to use data augmentation")

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

    # Set random seeds
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # make dirs
    if args.exp_name is None:
        args.exp_name = time.strftime("%Y%m%d-%H%M%S")

    if args.model_save_dir is not None:
        if not os.path.exists(args.model_save_dir):
            os.mkdir(args.model_save_dir)
        os.mkdir(os.path.join(args.model_save_dir, args.exp_name))

    if args.res_save_dir is not None:
        if not os.path.exists(args.res_save_dir):
            os.mkdir(args.res_save_dir)
        if os.path.exists(os.path.join(args.res_save_dir, args.exp_name)):
            respond = input("Experiment already exists, type [Y] to overwrite")
            if respond != "Y":
                exit(0)
        else:
            os.mkdir(os.path.join(args.res_save_dir, args.exp_name))

    assert args.res_save_dir is not None and args.exp_name is not None
    if args.dataset == "mnist17":
        if args.load_poison_dir is not None:
            data_loader = MNIST17DataPreprocessor.load(os.path.join(args.load_poison_dir, "data"), args)
        else:
            data_loader = MNIST17DataPreprocessor(args)
        model = MNIST17Model.MNIST17Model(data_loader.n_features, data_loader.n_classes, lr=args.lr)
    elif args.dataset == "mnist":
        if args.load_poison_dir is not None:
            data_loader = MNISTDataPreprocessor.load(os.path.join(args.load_poison_dir, "data"), args)
        else:
            data_loader = MNISTDataPreprocessor(args)
        model = MNISTModel.MNISTModel(data_loader.n_features, data_loader.n_classes, lr=args.lr)
    elif args.dataset == "imdb":
        if args.load_poison_dir is not None:
            data_loader = IMDBDataPreprocessor.load(os.path.join(args.load_poison_dir, "data"), args)
        else:
            data_loader = IMDBDataPreprocessor(args)
        model = IMDBTransformerModel.IMDBTransformerModel(data_loader.n_features, data_loader.n_classes, lr=args.lr)
    elif args.dataset in EMBER_DATASET:
        if args.load_poison_dir is not None:
            raise NotImplementedError  # todo: load data from file
            # data_loader = EmberDataPreProcessor.load(os.path.join(args.load_poison_dir, "data"), args)
        else:
            data_loader = EmberDataPreProcessor(args)
        model = EmberModel.EmberModel(data_loader.n_features, data_loader.n_classes, lr=args.lr)
    else:
        raise NotImplementedError

    aggregate_results = train_many(data_loader, model, args)
    np.save(os.path.join(args.res_save_dir, args.exp_name, "aggre_res"), aggregate_results)
    with open(os.path.join(args.res_save_dir, args.exp_name, "commandline_args.txt"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
