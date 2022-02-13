from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.layers import Conv2D, AveragePooling2D
from tensorflow.keras.regularizers import l2

from models.Model import Model


class MNISTModel(Model):
    def __init__(self, n_features, n_classes, lr=1e-3):
        super(MNISTModel, self).__init__(n_features, n_classes, lr)

    def build_model(self):
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
        return model
