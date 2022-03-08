from abc import ABC, abstractmethod
import os
import numpy as np

from tensorflow.keras.models import load_model


class Model(ABC):
    def __init__(self, input_shape, n_classes, lr):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.lr = lr
        self.model = self.build_model()
        self.weights_initialize = self.model.get_weights()

    @abstractmethod
    def build_model(self):
        pass

    def init(self):
        self.model.set_weights(self.weights_initialize)

    def save(self, save_path, file_name='mnist_nn'):
        save_model = self.model
        save_model.save(os.path.join(save_path, file_name + '.h5'))

    def load(self, save_path, file_name):
        self.model = load_model(os.path.join(save_path, file_name + '.h5'))

    def fit_generator(self, datagen, epochs):
        self.model.fit_generator(datagen, epochs=epochs, verbose=0, workers=4)

    def fit(self, X, y, batch_size, epochs):
        self.model.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=0, workers=4)

    def evaluate(self, x_test, y_test):
        score = self.model.evaluate(x_test, y_test, verbose=0, batch_size=512)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        prediction_label = np.argmax(self.model.predict(x_test), axis=1)
        return prediction_label
