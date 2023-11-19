import torch
import os
import json
import time
import tensorflow as tf
import numpy as np

from utils.data_processing import MNIST17DataPreprocessor, MNISTDataPreprocessor, IMDBDataPreprocessor, \
    EmberDataPreProcessor, EMBER_DATASET, EmberPoisonDataPreProcessor, MNIST01DataPreprocessor, \
    MNIST17LimitedDataPreprocessor, FMNISTDataPreprocessor, CIFARDataPreprocessor
from models import MNISTLiRPAModel, EmberLiRPAModel, EmberModel, MNISTModel, CIFAR10LiRPAModel, CIFAR10Model
from utils.train_utils import train_many
from utils.cert_train_argments import get_arguments, seed_everything

if __name__ == "__main__":
    parser = get_arguments()
    # Set random seeds
    args = parser.parse_args()
    seed_everything(args.seed)
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
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # make dirs
    if args.exp_name is None:
        args.exp_name = time.strftime("%Y%m%d-%H%M%S")

    if args.model_save_dir is not None:
        if not os.path.exists(args.model_save_dir):
            os.mkdir(args.model_save_dir)

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
    if args.dataset == "mnist":
        if args.load_poison_dir is not None:
            data_loader = MNISTDataPreprocessor.load(os.path.join(args.load_poison_dir, "data"), args)
        else:
            data_loader = MNISTDataPreprocessor(args)
        if args.no_lirpa:
            # use the same architecture as the LiRPAModel
            model = MNISTModel.MNIST17Model(data_loader.n_features, data_loader.n_classes, args.lr)
        else:
            model = MNISTLiRPAModel.MNISTModel(data_loader.n_features, data_loader.n_classes, args, device, lr=args.lr)
    elif args.dataset == "cifar10":
        if args.load_poison_dir is not None:
            data_loader = CIFARDataPreprocessor.load(os.path.join(args.load_poison_dir, "data"), args)
        else:
            data_loader = CIFARDataPreprocessor(args)
        if args.no_lirpa:
            # use the same architecture as the LiRPAModel
            model = CIFAR10Model.CIFAR10Model(data_loader.n_features, data_loader.n_classes, args.lr)
        else:
            model = CIFAR10LiRPAModel.CIFAR10Model(data_loader.n_features, data_loader.n_classes, args, device,
                                                   lr=args.lr)
    elif args.dataset == "ember":
        if args.load_poison_dir is not None:
            data_loader = EmberPoisonDataPreProcessor(args)
        else:
            data_loader = EmberDataPreProcessor(args)
        if args.no_lirpa:
            if args.select_strategy == "FPA":
                model = EmberModel.EmberModelOri(args.k, data_loader.n_classes, args.lr)
            else:
                model = EmberModel.EmberModel(data_loader.n_features, data_loader.n_classes, args.lr)
        else:
            model = EmberLiRPAModel.EmberModel(data_loader.n_features, data_loader.n_classes, args, device, lr=args.lr)
    else:
        raise NotImplementedError

    with open(os.path.join(args.res_save_dir, args.exp_name, "commandline_args.txt"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    aggregate_results = train_many(data_loader, model, args, res, res_noise)
