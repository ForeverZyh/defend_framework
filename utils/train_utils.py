import numpy as np
from tensorflow import keras
from tqdm import trange
import os
import gc

from utils.dataaug import DataGeneratorForMNIST, MNISTDataGenerator, EmberDataGenerator, CIFARDataGenerator
from utils import EMBER_DATASET, IMAGE_DATASET


def train_many(data_loader, model, args, aggregate_result, aggregate_noise_result):
    test_size = data_loader.x_test.shape[0]
    if aggregate_result is None:
        aggregate_result = np.zeros([test_size, data_loader.n_classes + 1 + int(args.select_strategy == "DPA")],
                                    dtype=np.int32)
        aggregate_noise_result = np.zeros([test_size, data_loader.n_classes + 1 + int(args.select_strategy == "DPA")],
                                          dtype=np.int32)
    aggregate_result[np.arange(0, test_size), -1] = data_loader.y_test
    aggregate_noise_result[np.arange(0, test_size), -1] = data_loader.y_test
    # set the DPA partition id
    data_loader.data_processor.DPA_partition_cnt = np.sum(aggregate_result[0, :-1])
    for i in trange(np.sum(aggregate_result[0, :-1]), args.N):
        datagen = None
        key_dict = {0: 0, 1: 1, 2: 2}  # used for imdb dataset to get word idx
        X, y = data_loader.data_processor.process_train(key_dict)
        # using the last index for the ground truth label
        y = keras.utils.to_categorical(y, data_loader.n_classes)
        X = np.repeat(X, args.stack_epochs, axis=0)
        y = np.repeat(y, args.stack_epochs, axis=0)
        if args.data_aug:
            if args.dataset in IMAGE_DATASET:
                if args.dataset == "cifar10":
                    datagen = CIFARDataGenerator(X, y, args.batch_size, data_loader.data_processor, args.no_eval_noise)
                else:
                    datagen = MNISTDataGenerator(X, y, args.batch_size, data_loader.data_processor, args.no_eval_noise)
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
        elif args.noise_strategy in ["RAB_gaussian", "RAB_uniform"] or (
                args.dataset in EMBER_DATASET and args.select_strategy in ["DPA", "FPA"]):
            x_test = data_loader.data_processor.minmax.transform(x_test)

        if datagen is not None:
            model.fit_generator(datagen, args.epochs)
        else:
            model.fit(X, y, args.batch_size, args.epochs) # , data_loader.data_processor.process_test(x_test, args.fix_noise), y_test)

        if args.select_strategy not in ["DPA", "FPA"]:
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
        elif args.select_strategy == "DPA":
            if args.patchguard:
                prediction_label, conf = model.evaluate(x_test, y_test)
                aggregate_result[np.arange(0, test_size), prediction_label] += 1
                aggregate_noise_result[np.arange(0, test_size), prediction_label] += 1
                prediction_label = prediction_label, conf
            elif args.no_lirpa:
                prediction_label = model.evaluate(x_test, y_test)
                aggregate_result[np.arange(0, test_size), prediction_label] += 1
                aggregate_noise_result[np.arange(0, test_size), prediction_label] += 1
            else:
                prediction_label, prediction_label_cert = model.evaluate(x_test, y_test)
                aggregate_result[np.arange(0, test_size), prediction_label] += 1
                aggregate_noise_result[np.arange(0, test_size), prediction_label_cert] += 1
        elif args.select_strategy == "FPA":
            assert args.no_lirpa
            x_test = data_loader.data_processor.process_test(x_test, args.fix_noise)
            prediction_label = model.evaluate(x_test, y_test)
            aggregate_result[np.arange(0, test_size), prediction_label] += 1
            aggregate_noise_result[np.arange(0, test_size), prediction_label] += 1
        else:
            raise NotImplementedError

        if args.model_save_dir is not None:
            model.save(args.model_save_dir, str(i),
                       prediction_label if args.SABR or args.patchguard else None)
        model.init()
        if i % 50 == 0:
            np.save(os.path.join(args.res_save_dir, args.exp_name, "aggre_res"),
                    (aggregate_result, aggregate_noise_result))
        gc.collect()

    np.save(os.path.join(args.res_save_dir, args.exp_name, "aggre_res"), (aggregate_result, aggregate_noise_result))
    print(aggregate_result, aggregate_noise_result)


def train_single(data_loader, model, args):
    # train single classifier for attacking
    model.fit(data_loader.x_train, keras.utils.to_categorical(data_loader.y_train, data_loader.n_classes),
              args.batch_size, args.epochs)
    # model.save(args.model_save_dir)
