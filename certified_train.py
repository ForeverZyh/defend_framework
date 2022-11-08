import random
import os
import json
import time

import numpy as np
import tensorflow as tf

from utils.data_processing import MNIST17DataPreprocessor, MNISTDataPreprocessor, IMDBDataPreprocessor, \
    EmberDataPreProcessor, EMBER_DATASET, EmberPoisonDataPreProcessor, MNIST01DataPreprocessor, \
    MNIST17LimitedDataPreprocessor, FMNISTDataPreprocessor, CIFARDataPreprocessor, ContagioDataPreProcessor, \
    CIFAR02DataPreprocessor
from models import MNISTModel, IMDBTransformerModel, EmberModel, CIFAR10Model, ContagioModel
from utils.train_utils import train_many
from utils.cert_train_argments import get_arguments

if __name__ == "__main__":
    parser = get_arguments()
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

    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # tf.random.set_seed(args.seed)

    # make dirs
    if args.exp_name is None:
        args.exp_name = time.strftime("%Y%m%d-%H%M%S")

    if args.model_save_dir is not None:
        if not os.path.exists(args.model_save_dir):
            os.mkdir(args.model_save_dir)
        os.mkdir(os.path.join(args.model_save_dir, args.exp_name))
        args.model_save_dir = os.path.join(args.model_save_dir, args.exp_name)

    res = None
    res_noise = None
    if args.res_save_dir is not None:
        if not os.path.exists(args.res_save_dir):
            os.mkdir(args.res_save_dir)
        if os.path.exists(os.path.join(args.res_save_dir, args.exp_name)):
            respond = input("Experiment already exists, type [O] to overwrite, type [R] to resume")
            if respond == "O":
                pass
            elif respond == "R":
                res, res_noise = np.load(os.path.join(args.res_save_dir, args.exp_name, "aggre_res.npy"))
            else:
                exit(0)
        else:
            os.mkdir(os.path.join(args.res_save_dir, args.exp_name))

    assert args.res_save_dir is not None and args.exp_name is not None
    if args.dataset == "mnist17":
        if args.load_poison_dir is not None:
            data_loader = MNIST17DataPreprocessor.load(os.path.join(args.load_poison_dir, "data"), args)
        else:
            data_loader = MNIST17DataPreprocessor(args)
        model = MNISTModel.MNIST17Model(data_loader.n_features, data_loader.n_classes, lr=args.lr)
    elif args.dataset == "mnist17_limited":
        if args.load_poison_dir is not None:
            data_loader = MNIST17LimitedDataPreprocessor.load(os.path.join(args.load_poison_dir, "data"), args)
        else:
            data_loader = MNIST17LimitedDataPreprocessor(args)
        model = MNISTModel.MNIST17Model(data_loader.n_features, data_loader.n_classes, lr=args.lr)
    elif args.dataset == "mnist01":
        if args.load_poison_dir is not None:
            data_loader = MNIST01DataPreprocessor.load(os.path.join(args.load_poison_dir, "data"), args)
        else:
            data_loader = MNIST01DataPreprocessor(args)
        model = MNISTModel.MNIST01Model(data_loader.n_features, data_loader.n_classes, lr=args.lr)
    elif args.dataset == "mnist":
        if args.load_poison_dir is not None:
            data_loader = MNISTDataPreprocessor.load(os.path.join(args.load_poison_dir, "data"), args)
        else:
            data_loader = MNISTDataPreprocessor(args)
        model = MNISTModel.MNISTModel(data_loader.n_features, data_loader.n_classes, lr=args.lr)
    elif args.dataset == "fmnist":
        if args.load_poison_dir is not None:
            data_loader = FMNISTDataPreprocessor.load(os.path.join(args.load_poison_dir, "data"), args)
        else:
            data_loader = FMNISTDataPreprocessor(args)
        model = MNISTModel.FMNISTModel(data_loader.n_features, data_loader.n_classes, lr=args.lr)
    elif args.dataset == "cifar10":
        if args.load_poison_dir is not None:
            data_loader = CIFARDataPreprocessor.load(os.path.join(args.load_poison_dir, "data"), args)
        else:
            data_loader = CIFARDataPreprocessor(args)
        model = CIFAR10Model.CIFAR10Model(data_loader.n_features, data_loader.n_classes, lr=args.lr)
    elif args.dataset == "cifar10-02":
        if args.load_poison_dir is not None:
            data_loader = CIFAR02DataPreprocessor.load(os.path.join(args.load_poison_dir, "data"), args)
        else:
            data_loader = CIFAR02DataPreprocessor(args)
        model = CIFAR10Model.CIFAR10Model(data_loader.n_features, data_loader.n_classes, lr=args.lr)
    elif args.dataset == "imdb":
        if args.load_poison_dir is not None:
            data_loader = IMDBDataPreprocessor.load(os.path.join(args.load_poison_dir, "data"), args)
        else:
            data_loader = IMDBDataPreprocessor(args)
        model = IMDBTransformerModel.IMDBTransformerModel(data_loader.n_features, data_loader.n_classes, lr=args.lr)
    elif args.dataset in EMBER_DATASET:
        if args.load_poison_dir is not None:
            data_loader = EmberPoisonDataPreProcessor(args)
        else:
            data_loader = EmberDataPreProcessor(args)
        model = EmberModel.EmberModel(data_loader.n_features, data_loader.n_classes, lr=args.lr)
    elif args.dataset == "contagio":
        if args.load_poison_dir is not None:
            raise NotImplementedError
        else:
            data_loader = ContagioDataPreProcessor(args)
        model = ContagioModel.ContagioModel(data_loader.n_features, data_loader.n_classes, lr=args.lr)
    else:
        raise NotImplementedError

    with open(os.path.join(args.res_save_dir, args.exp_name, "commandline_args.txt"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    aggregate_results = train_many(data_loader, model, args, res, res_noise)
