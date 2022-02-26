from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization, Activation
from tensorflow.keras.layers import Conv2D, AveragePooling2D
from tensorflow.keras.regularizers import l2

from models.Model import Model


class EmberModel(Model):
    def __init__(self, n_features, n_classes, lr=1e-3):
        super(EmberModel, self).__init__(n_features, n_classes, lr)

    def build_model(self):
        model = Sequential()
        model.add(Input(shape=(self.input_shape,)))
        model.add(Dense(2000, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(1000, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(100, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))
        momentum = 0.9
        decay = 0.000001
        opt = keras.optimizers.SGD(lr=self.lr, momentum=momentum, decay=decay)

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=opt,
                      metrics=['accuracy'])
        return model
