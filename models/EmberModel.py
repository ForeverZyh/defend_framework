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
        model.Add(Input(shape=(self.input_shape,)))
        model.Add(Dense(2000, activation='relu'))
        model.Add(BatchNormalization())
        model.Add(Dropout(0.5))
        model.Add(Dense(1000, activation='relu'))
        model.Add(BatchNormalization())
        model.Add(Dropout(0.5))
        model.Add(Dense(100, activation='relu'))
        model.Add(BatchNormalization())
        model.Add(Dropout(0.5))
        model.Add(Dense(1, activation='sigmoid'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(lr=self.lr),
                      metrics=['accuracy'])
        return model
