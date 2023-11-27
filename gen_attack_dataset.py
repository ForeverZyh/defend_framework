import random
import os
import json
import time
import torch
import numpy as np
import tensorflow as tf
from tensorflow import keras
import warnings
import wandb

from utils.data_processing import MNISTDataPreprocessor, MNIST17DataPreprocessor, MNIST01DataPreprocessor, \
    CIFAR02DataPreprocessor
from models.MNISTModel import MNISTModel, MNIST17Model, MNIST01Model
from models.CIFAR10Model import CIFAR10Model
from models.bagnet import BagNetModel
from utils.train_utils import train_single
from attack.BadNetAttack import BadNetAttackLabel, BadNetAttackNoLabel
from utils.cert_train_argments import get_arguments, seed_everything

# silence ResourceWarning
warnings.filterwarnings("ignore", category=ResourceWarning)

if __name__ == "__main__":
    parser = get_arguments()
    # attack parameters
    parser.add_argument("--attack", choices=["badnet"], help="attack algorithms", required=True)
    parser.add_argument("--consecutive", action="store_true",
                        help="Whether the poisoned features need to be inside a block")
    parser.add_argument("--attack_label", action="store_true",
                        help="Whether to attack the label of the training image")
    parser.add_argument("--poisoned_feat_num", type=int, required=True, help="poisoned feature number")
    parser.add_argument("--poisoned_ins_rate", default=0.1, type=float, help="the rate of instances to be poisoned")
    parser.add_argument("--attack_targets", type=str,
                        help="A list of ints of length n_classes, attacking label i to its target attack_targets[i], "
                             "attack_targets[i] can be None.")

    # dirs and files
    parser.add_argument("--save_poison_dir", type=str,
                        help="dir for save poisoned dataset"
                        )
    parser.add_argument("--load", action="store_true", help="whether to load the saved file")
    # parser.add_argument("--tau", default=0.5, type=float, help="the parameter tau in the defense")

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
        if args.patchguard:
            Model_type = BagNetModel
        else:
            Model_type = MNISTModel
    elif args.dataset == "mnist17":
        DataPreprocessor_type = MNIST17DataPreprocessor
        Model_type = MNIST17Model
    elif args.dataset == "mnist01":
        DataPreprocessor_type = MNIST01DataPreprocessor
        Model_type = MNIST01Model
    elif args.dataset == "cifar10-02":
        DataPreprocessor_type = CIFAR02DataPreprocessor
        if args.patchguard:
            Model_type = BagNetModel
        else:
            Model_type = CIFAR10Model
    else:
        raise NotImplementedError

    with open(os.path.join(filepath, "commandline_args.txt"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    attack_targets = eval(args.attack_targets)
    if not args.load:
        data_loader = DataPreprocessor_type(args)
        if args.attack_label:
            attack = BadNetAttackLabel(data_loader, attack_targets, args.poisoned_feat_num,
                                       consecutive=args.consecutive, poisoned_ins_rate=args.poisoned_ins_rate)
        else:
            attack = BadNetAttackNoLabel(data_loader, attack_targets, args.poisoned_feat_num,
                                         consecutive=args.consecutive, poisoned_ins_rate=args.poisoned_ins_rate)
        attack.attack()
        attack.save(os.path.join(filepath, "data"))
    else:
        if args.attack_label:
            attack = BadNetAttackLabel.load(os.path.join(filepath, "data"))
        else:
            attack = BadNetAttackNoLabel.load(os.path.join(filepath, "data"))
        data_loader = attack.data_processor

    if args.wandb:
        args.wandb = wandb.init(project="poison_defense", name=args.exp_name, config=args.__dict__)
    else:
        args.wandb = None
    if args.patchguard:
        model = Model_type(data_loader.n_features, data_loader.n_classes, lr=args.lr, device=device,
                           patch_size=args.poisoned_feat_num, weight_decay=args.weight_decay,
                           x_test=data_loader.x_test, y_test=data_loader.y_test, wandb=args.wandb,
                           pretrained=True)
    else:
        model = Model_type(data_loader.n_features, data_loader.n_classes)

    train_single(data_loader, model, args)
    print("Clean Test Set:")
    res = model.evaluate(data_loader.x_test, keras.utils.to_categorical(data_loader.y_test, data_loader.n_classes))
    print("Poisoned Test Set:")
    res1 = model.evaluate(data_loader.x_test_poisoned,
                          keras.utils.to_categorical(data_loader.y_test, data_loader.n_classes))
    model.save(filepath, predictions=res + res1)
    for i in range(data_loader.n_classes):
        idx = np.where(data_loader.y_test == i)[0]
        if attack_targets[i] is None:
            print(f"class {i} is not poisoned:")
            model.evaluate(data_loader.x_test_poisoned[idx],
                           keras.utils.to_categorical(data_loader.y_test_poisoned[idx], data_loader.n_classes),
                           tune=False)
        else:
            print(f"class {i} is poisoned:")
            model.evaluate(data_loader.x_test_poisoned[idx],
                           keras.utils.to_categorical(data_loader.y_test_poisoned[idx], data_loader.n_classes),
                           tune=False)
