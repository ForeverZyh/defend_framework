from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
import numpy as np
import torch
from torchvision import transforms


def DataGeneratorForMNIST():
    datagen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # epsilon for ZCA whitening
        zca_epsilon=1e-06,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=10,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # set range for random shear
        shear_range=0.0,
        # set range for random zoom
        zoom_range=0.0,
        # set range for random channel shifts
        channel_shift_range=0.,
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        # value used for fill_mode = "constant"
        cval=0.,
        # randomly flip images
        horizontal_flip=False,
        # randomly flip images
        vertical_flip=False,
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format="channels_last",
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)
    return datagen


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, X, y, batch_size, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.X = X
        self.y = y
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        return self.X[indexes], self.y[indexes]

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(self.indexes)
        self.get_random()

    def get_random(self):
        pass


class MNISTDataGenerator(DataGenerator):
    def __init__(self, X, y, batch_size, data_processor, no_eval_noise, shuffle=True):
        self.data_processor = data_processor
        self.no_eval_noise = no_eval_noise
        super(MNISTDataGenerator, self).__init__(X, y, batch_size, shuffle=shuffle)

    def get_random(self):
        data = torch.Tensor(np.transpose(self.X, (0, 3, 1, 2)))
        data = transforms.RandomCrop(28, 2)(data)
        data = transforms.RandomRotation(10)(data)
        data = np.transpose(data.numpy(), (0, 2, 3, 1))
        self.data_aug = data
        self.data_noise = self.data_processor.noise_data(self.X)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        X, y = super(MNISTDataGenerator, self).__getitem__(index)
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        data_aug = self.data_aug[indexes]
        data_noise = self.data_noise[indexes]
        if self.no_eval_noise:
            return np.vstack([data_aug, X]), np.vstack([y, y])
            # return np.vstack([X + np.random.normal(0, 0.1, X.shape), X]), np.vstack([y, y])
        else:
            return np.vstack([data_noise, data_aug, X]), np.vstack([y, y, y])


class CIFARDataGenerator(DataGenerator):
    def __init__(self, X, y, batch_size, data_processor, no_eval_noise, shuffle=True):
        self.data_processor = data_processor
        self.no_eval_noise = no_eval_noise
        super(CIFARDataGenerator, self).__init__(X, y, batch_size, shuffle=shuffle)

    def get_random(self):
        data = torch.Tensor(np.transpose(self.X, (0, 3, 1, 2)))
        data = transforms.RandomCrop(32, 3)(data)
        data = transforms.RandomRotation(10)(data)
        data = np.transpose(data.numpy(), (0, 2, 3, 1))
        self.data_aug = data
        self.data_noise = self.data_processor.noise_data(self.X)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        X, y = super(CIFARDataGenerator, self).__getitem__(index)
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        data_aug = self.data_aug[indexes]
        data_noise = self.data_noise[indexes]
        if self.no_eval_noise:
            return np.vstack([data_aug, X]), np.vstack([y, y])
            # return np.vstack([X + np.random.normal(0, 0.1, X.shape), X]), np.vstack([y, y])
        else:
            return np.vstack([data_noise, data_aug, X]), np.vstack([y, y, y])


class EmberDataGenerator(DataGenerator):
    def __init__(self, X, y, batch_size, data_processor, no_eval_noise, shuffle=True):
        self.data_processor = data_processor
        self.no_eval_noise = no_eval_noise
        super(EmberDataGenerator, self).__init__(X, y, batch_size, shuffle=shuffle)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        X, y = super(EmberDataGenerator, self).__getitem__(index)
        if self.no_eval_noise:
            return np.vstack([X + np.random.normal(0, 0.1, X.shape), X]), np.vstack([y, y])
        else:
            return np.vstack([self.data_processor.noise_data(X), X + np.random.normal(0, 0.1, X.shape), X]), np.vstack(
                [y, y, y])
