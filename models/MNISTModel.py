import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv2D

from models.Model import Model


class MNISTModel(Model):
    def __init__(self, n_features, n_classes):
        super(MNISTModel, self).__init__(n_features, n_classes)

    def build_model(self, input_shape, n_classes):
        model = Sequential()
        model.add(Conv2D(64, kernel_size=(4, 4),
                         activation='relu',
                         input_shape=input_shape,
                         strides=2))
        model.add(Conv2D(128, (4, 4), activation='relu', strides=2))
        model.add(Flatten())
        model.add(Dense(500, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(n_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(lr=0.01, decay=0.98),
                      metrics=['accuracy'])
        return model
