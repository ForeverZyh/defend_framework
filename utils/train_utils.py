import numpy as np
from tensorflow import keras

from utils.dataaug import DataGeneratorForMNIST


def train_many(data_loader, model, args):
    test_size = data_loader.x_test.shape[0]
    aggregate_result = np.zeros([test_size, data_loader.n_classes + 1], dtype=np.int)
    aggregate_noise_result = np.zeros([test_size, data_loader.n_classes + 1], dtype=np.int)
    # using the last index for the ground truth label
    datagen = DataGeneratorForMNIST() if args.data_aug and args.dataset in ["mnist", "mnist17"] else None
    for i in range(args.N):
        key_dict = {0: 0, 1: 1, 2: 2}  # used for imdb dataset to get word idx
        X, y = data_loader.data_processor.process_train(key_dict)
        y = keras.utils.to_categorical(y, data_loader.n_classes)
        if datagen is not None:
            model.fit_generator(datagen.flow(X, y, batch_size=args.batch_size), args.epochs)
        else:
            model.fit(X, y, args.batch_size, args.epochs)
        x_test = data_loader.x_test.copy()
        if args.dataset == "imdb":
            for x in x_test:
                for i in range(len(x)):
                    if x[i] in key_dict:
                        x[i] = key_dict[x[i]]
                    else:
                        x[i] = 2
        elif args.dataset == "ember" and args.noise_strategy in ["feature_flipping", "all_flipping"]:
            x_test = data_loader.data_processor.kbin.transform(x_test)

        prediction_label = model.evaluate(x_test,
                                          keras.utils.to_categorical(data_loader.y_test, data_loader.n_classes))
        aggregate_result[np.arange(0, test_size), prediction_label] += 1
        X_test = data_loader.data_processor.process_test(x_test, False)
        prediction_label = model.evaluate(X_test, keras.utils.to_categorical(data_loader.y_test, data_loader.n_classes))
        aggregate_noise_result[np.arange(0, test_size), prediction_label] += 1

        model.init()
    aggregate_result[np.arange(0, test_size), -1] = data_loader.y_test
    aggregate_noise_result[np.arange(0, test_size), -1] = data_loader.y_test
    print(aggregate_result, aggregate_noise_result)
    return aggregate_result, aggregate_noise_result


def train_single(data_loader, model, args):
    # train single classifier for attacking
    model.fit(data_loader.x_train, keras.utils.to_categorical(data_loader.y_train, data_loader.n_classes),
              args.batch_size, args.epochs)
    # model.save(args.model_save_dir)
