import numpy as np
from tensorflow import keras

from dataaug import DataGeneratorForMNIST


def train_many(data_loader, model, args):
    test_size = data_loader.x_test.shape[0]
    aggregate_result = np.zeros([test_size, data_loader.n_classes + 1], dtype=np.int)
    # using the last index for the ground truth label
    datagen = DataGeneratorForMNIST() if args.data_aug else None
    for i in range(args.N):
        X, y = data_loader.data_processor.process_train()
        y = keras.utils.to_categorical(y, data_loader.n_classes)
        if datagen is not None:
            model.fit_generator(datagen.flow(X, y, batch_size=args.batch_size), args.epochs)
        else:
            model.fit(X, y, args.batch_size, args.epochs)
        prediction_label = model.evaluate(data_loader.x_test,
                                          keras.utils.to_categorical(data_loader.y_test, data_loader.n_classes))
        aggregate_result[np.arange(0, test_size), prediction_label] += 1
        model.init()
    aggregate_result[np.arange(0, test_size), -1] = data_loader.y_test
    print(aggregate_result)
    return aggregate_result


def train_single(data_loader, model, args):
    # train single classifier for attacking
    model.fit(data_loader.x_train, keras.utils.to_categorical(data_loader.y_train, data_loader.n_classes),
              args.batch_size, args.epochs)
    # model.save(args.model_save_dir)
