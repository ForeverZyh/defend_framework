import os
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model


class MNISTModel(object):
    def __init__(self, n_features, n_classes):
        self.n_features = n_features
        self.model = self.build_model(n_features, n_classes)

    def build_model(self, input_shape, n_classes):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=input_shape))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(n_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        return model

    def save(self, save_path, file_name='mnist_nn'):
        save_model = self.model
        save_model.save(os.path.join(save_path, file_name + '.h5'))

    def load(self, save_path, file_name):
        self.model = load_model(os.path.join(save_path, file_name + '.h5'))

    def fit_generator(self, datagen, epochs):
        self.model.fit_generator(datagen, epochs=epochs, verbose=0, workers=4)

    def fit(self, X, y, batch_size, epochs):
        self.model.fit(X, y, batch_size=batch_size, epochs=epochs)

    def evaluate(self, x_test, y_test):
        score = self.model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        prediction_label = np.argmax(self.model.predict(x_test), axis=1)
        return prediction_label
