from abc import ABC, abstractmethod
import os
import numpy as np
from keras.models import load_model


class Model(ABC):
    def __init__(self, n_features, n_classes, lr):
        self.n_features = n_features
        self.lr = lr
        self.model = self.build_model(n_features, n_classes)
        self.weights_initialize = self.model.get_weights()

    @abstractmethod
    def build_model(self, n_features, n_classes):
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
        score = self.model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        prediction_label = np.argmax(self.model.predict(x_test), axis=1)
        return prediction_label
