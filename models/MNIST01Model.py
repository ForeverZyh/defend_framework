from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

from models.Model import Model


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