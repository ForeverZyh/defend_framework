import numpy as np
from tensorflow import keras
from tqdm import trange
import os

from utils.dataaug import DataGeneratorForMNIST, MNISTDataGenerator, EmberDataGenerator
from utils import EMBER_DATASET, IMAGE_DATASET


def train_many(data_loader, model, args, aggregate_result, aggregate_noise_result):
    test_size = data_loader.x_test.shape[0]
    if aggregate_result is None:
        aggregate_result = np.zeros([test_size, data_loader.n_classes + 1], dtype=np.int)
        aggregate_noise_result = np.zeros([test_size, data_loader.n_classes + 1], dtype=np.int)
    aggregate_result[np.arange(0, test_size), -1] = data_loader.y_test
    aggregate_noise_result[np.arange(0, test_size), -1] = data_loader.y_test

    remaining = args.N - np.sum(aggregate_result[0, :-1])
    datagen = None
    for i in trange(remaining):
        key_dict = {0: 0, 1: 1, 2: 2}  # used for imdb dataset to get word idx
        X, y = data_loader.data_processor.process_train(key_dict)
        # using the last index for the ground truth label
        y = keras.utils.to_categorical(y, data_loader.n_classes)
        if args.data_aug:
            if args.dataset in IMAGE_DATASET:
                datagen = MNISTDataGenerator(X, y, args.batch_size)
            elif args.dataset in EMBER_DATASET:
                datagen = EmberDataGenerator(X, y, args.batch_size, data_loader.data_processor, args.no_eval_noise)
        y_test = keras.utils.to_categorical(data_loader.y_test, data_loader.n_classes)
        x_test = data_loader.x_test.copy()
        if args.dataset == "imdb":
            for x in x_test:
                for i in range(len(x)):
                    if x[i] in key_dict:
                        x[i] = key_dict[x[i]]
                    else:
                        x[i] = 2
        elif args.dataset in EMBER_DATASET and args.noise_strategy in ["feature_flipping", "all_flipping"]:
            categorized = data_loader.data_processor.kbin.transform(x_test) / args.K
            if args.dataset == "ember_limited":
                x_test[:, data_loader.data_processor.limit_id] = categorized[:, data_loader.data_processor.limit_id]
            else:
                x_test = categorized
        elif args.noise_strategy in ["RAB_gaussian", "RAB_uniform"]:
            x_test = data_loader.data_processor.minmax.transform(x_test)

        if datagen is not None:
            model.fit_generator(datagen, args.epochs)
        else:
            model.fit(X, y, args.batch_size, args.epochs)

        if args.dataset in EMBER_DATASET and args.noise_strategy is None:
            prediction_label = model.evaluate(data_loader.data_processor.normal.transform(x_test), y_test)
        else:
            prediction_label = model.evaluate(x_test, y_test)
        aggregate_result[np.arange(0, test_size), prediction_label] += 1
        if args.noise_strategy is None or args.no_eval_noise:
            aggregate_noise_result[np.arange(0, test_size), prediction_label] += 1
        else:
            X_test = data_loader.data_processor.process_test(x_test, args.fix_noise)
            prediction_label = model.evaluate(X_test, y_test)
            aggregate_noise_result[np.arange(0, test_size), prediction_label] += 1

        model.init()
        np.save(os.path.join(args.res_save_dir, args.exp_name, "aggre_res"), (aggregate_result, aggregate_noise_result))

    print(aggregate_result, aggregate_noise_result)


def train_single(data_loader, model, args):
    # train single classifier for attacking
    model.fit(data_loader.x_train, keras.utils.to_categorical(data_loader.y_train, data_loader.n_classes),
              args.batch_size, args.epochs)
    # model.save(args.model_save_dir)
