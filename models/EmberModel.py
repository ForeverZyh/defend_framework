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
        def scheduler(epoch, lr):
            if epoch > 0 and epoch % 10 == 0:
                lr *= 0.9
            # print(lr)
            return lr

        reg = l2(1e-5)
        model = Sequential()
        model.add(Input(shape=(self.input_shape,)))
        model.add(Dense(2000, activation='relu', kernel_regularizer=reg, bias_regularizer=reg))
        # model.add(BatchNormalization())
        model.add(Dropout(0.25))
        model.add(Dense(1000, activation='relu', kernel_regularizer=reg, bias_regularizer=reg))
        # model.add(BatchNormalization())
        model.add(Dropout(0.25))
        model.add(Dense(100, activation='relu', kernel_regularizer=reg, bias_regularizer=reg))
        # model.add(BatchNormalization())
        model.add(Dropout(0.25))
        model.add(Dense(2, activation='softmax'))
        momentum = 0.9
        decay = 0.000001
        opt = keras.optimizers.SGD(lr=self.lr, momentum=momentum, decay=decay)
        self.callback = []
        self.callback += [keras.callbacks.LearningRateScheduler(scheduler)]
        # self.callback += [keras.callbacks.EarlyStopping(
        #     monitor="loss",
        #     min_delta=0.01,
        #     patience=20,
        #     verbose=0,
        #     baseline=None,
        #     restore_best_weights=True,
        # )]

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(lr=self.lr),
                      metrics=['accuracy'],
                      )
        return model
