import argparse
import random
import os
import json
import time

import numpy as np
import tensorflow as tf
import keras

from utils.data_processing import MNISTDataPreprocessor, MNIST17DataPreprocessor
from models.MNISTModel import MNISTModel
from models.MNIST17Model import MNIST17Model
from utils.train_utils import train_single
from attack.BadNetAttack import BadNetAttack

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # general training parameters
    parser.add_argument("-d", "--dataset", choices=["ember", "mnist", "mnist17"],
                        help="dataset type", required=True)
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--epochs", type=int, default=200, help="training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu id for training")
    parser.add_argument("--data_aug", action='store_true', help="whether to use data augmentation")

    # attack parameters
    parser.add_argument("--attack", choices=["badnet"], help="attack algorithms", required=True)
    parser.add_argument("--consecutive", action="store_true",
                        help="Whether the poisoned features need to be inside a block")
    parser.add_argument("--poisoned_feat_num", type=int, required=True, help="poisoned feature number")
    parser.add_argument("--poisoned_ins_rate", default=0.1, type=float, help="the rate of instances to be poisoned")
    parser.add_argument("--attack_targets", type=str,
                        help="A list of ints of length n_classes, attacking label i to its target attack_targets[i], "
                             "attack_targets[i] can be None.")

    # poisoning defence parameters
    parser.add_argument('--k', action='store', default=None, type=int,
                        help='number of (expected) examples in a bag')
    parser.add_argument("--select_strategy", default=None,
                        choices=["bagging_replace", "bagging_wo_replace", "binomial"],
                        help="selection strategy")
    parser.add_argument("--noise_strategy", default=None,
                        choices=["feature_flipping", "label_flipping", "all_flipping", "RAB_gaussian", "RAB_uniform"],
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

    # dirs and files
    parser.add_argument("--save_poison_dir", type=str,
                        help="dir for save poisoned dataset"
                        )
    parser.add_argument("--load", action="store_true", help="whether to load the saved file")
    parser.add_argument("--exp_name", default=None, type=str, help="name for this experiment")

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
    if not os.path.exists(args.save_poison_dir):
        os.mkdir(args.save_poison_dir)

    if args.exp_name is None:
        args.exp_name = time.strftime("%Y%m%d-%H%M%S")
    filepath = os.path.join(args.save_poison_dir, args.exp_name)

    if args.load:
        assert os.path.exists(filepath) and os.path.exists(os.path.join(filepath, "data"))

    if os.path.exists(filepath) and not args.load:
        respond = input("Experiment already exists, type [Y] to overwrite")
        if respond != "Y":
            exit(0)
    elif not os.path.exists(filepath):
        os.mkdir(filepath)

    if args.dataset == "mnist":
        DataPreprocessor_type = MNISTDataPreprocessor
        Model_type = MNISTModel
    elif args.dataset == "mnist17":
        DataPreprocessor_type = MNIST17DataPreprocessor
        Model_type = MNIST17Model
    else:
        raise NotImplementedError

    with open(os.path.join(filepath, "commandline_args.txt"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    attack_targets = eval(args.attack_targets)
    if not args.load:
        data_loader = DataPreprocessor_type(args)
        attack = BadNetAttack(data_loader, attack_targets, args.poisoned_feat_num,
                              consecutive=args.consecutive, poisoned_ins_rate=args.poisoned_ins_rate)
        attack.attack()
        attack.save(os.path.join(filepath, "data"))
    else:
        attack = BadNetAttack.load(os.path.join(filepath, "data"))
        data_loader = attack.data_processor

    model = Model_type(data_loader.n_features, data_loader.n_classes)
    train_single(data_loader, model, args)
    print("Clean Test Set:")
    model.evaluate(data_loader.x_test, keras.utils.to_categorical(data_loader.y_test, data_loader.n_classes))
    print("Poisoned Test Set:")
    for i in range(data_loader.n_classes):
        idx = np.where(data_loader.y_test == i)[0]
        if attack_targets[i] is None:
            print(f"class {i} is not poisoned:")
            model.evaluate(data_loader.x_test_poisoned[idx],
                           keras.utils.to_categorical(data_loader.y_test_poisoned[idx], data_loader.n_classes))
        else:
            print(f"class {i} is poisoned:")
            model.evaluate(data_loader.x_test_poisoned[idx],
                           keras.utils.to_categorical(data_loader.y_test_poisoned[idx], data_loader.n_classes))
