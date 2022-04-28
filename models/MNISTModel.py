from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D
from tensorflow.keras.regularizers import l2

from models.Model import Model


class MNISTModel(Model):
    def __init__(self, n_features, n_classes, lr=1e-3):
        super(MNISTModel, self).__init__(n_features, n_classes, lr)

    def build_model(self):
        def scheduler(epoch, lr):
            if epoch > 0 and epoch % 100 == 0:
                lr *= 0.5
            # print(lr)
            return lr

        reg = l2(1e-3)
        model = Sequential()
        model.add(Conv2D(16, kernel_size=(5, 5),
                         activation='relu',
                         input_shape=self.input_shape))
        model.add(AveragePooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, (5, 5), activation='relu'))
        model.add(AveragePooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dropout(0.25))
        model.add(Dense(32, activation='relu', kernel_regularizer=reg, bias_regularizer=reg))
        model.add(Dropout(0.25))
        model.add(Dense(512, activation='relu', kernel_regularizer=reg, bias_regularizer=reg))
        model.add(Dense(self.n_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(lr=self.lr),
                      metrics=['accuracy'])
        # self.callback = [keras.callbacks.LearningRateScheduler(scheduler)]
        return model


class MNIST01Model(Model):
    def __init__(self, n_features, n_classes, lr=1e-3):
        super(MNIST01Model, self).__init__(n_features, n_classes, lr)

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(16, kernel_size=(5, 5),
                         activation='relu',
                         input_shape=self.input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        model.add(Conv2D(32, (5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.n_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(lr=self.lr),
                      metrics=['accuracy'])
        return model


class MNIST17Model(Model):
    def __init__(self, n_features, n_classes, lr=1e-3):
        super(MNIST17Model, self).__init__(n_features, n_classes, lr)

    def build_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.input_shape))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.n_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(lr=self.lr),
                      metrics=['accuracy'])
        return model


class FMNISTModel(Model):
    def __init__(self, n_features, n_classes, lr=1e-3):
        super(FMNISTModel, self).__init__(n_features, n_classes, lr)

    def build_model(self):
        reg = l2(1e-3)
        model = Sequential()
        model.add(Conv2D(16, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=self.input_shape))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        model.add(Flatten())
        model.add(Dropout(0.6))
        model.add(Dense(128, activation='relu', kernel_regularizer=reg, bias_regularizer=reg))
        model.add(Dropout(0.5))
        model.add(Dense(self.n_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(lr=self.lr),
                      metrics=['accuracy'])
        return model
