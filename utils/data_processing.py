import os
import pickle

import numpy as np
from tensorflow.keras.datasets import mnist, imdb
from tensorflow import keras
import ember
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer

from utils.ember_feature_utils import load_features

EMBER_DATASET = ["ember", "ember_limited"]
FEATURE_DATASET = ["mnist", "mnist17"] + EMBER_DATASET
LANGUAGE_DATASET = ["imdb"]


class DataProcessor:
    def __init__(self, X, y, select_strategy=None, k=None, noise_strategy=None, dataset=None, **kwargs):
        """
        The initializer of data processor
        :param X: the training data (features)
        :param y: the training labels
        :param select_strategy: ["bagging_replace", "bagging_wo_replace", "binomial"]
            bagging_replace: bagging with replacement (the original bagging paper)
            bagging_wo_replace: bagging without replacement
            binomial: select each instance with probability p = k / |X|
        :param k: the size of the (expected) bag
        :param noise_strategy: ["feature_flipping", "label_flipping", "RAB_gaussian", "RAB_uniform"]
            feature_flipping / label_flipping: each feature / label remains with alpha, flipped with 1 - alpha
            RAB_gaussian: add gaussian noise of mu=0, sigma
            RAB_uniform: add uniform noise of U[a, b]
        :param dataset: the name of the dataset
        :param kwargs: the parameters for each noise strategy
        """
        self.select_strategy = select_strategy
        self.noise_strategy = noise_strategy
        self.dataset = dataset
        self.X = X
        self.y = y
        if select_strategy is not None:
            assert select_strategy in ["bagging_replace", "bagging_wo_replace", "binomial"]
            assert 0 < k <= len(X)
            self.k = k

        if noise_strategy is not None:
            assert noise_strategy in ["feature_flipping", "label_flipping", "all_flipping", "RAB_gaussian",
                                      "RAB_uniform", "sentence_select"]
            if dataset in FEATURE_DATASET:
                if noise_strategy in ["feature_flipping", "label_flipping", "all_flipping"]:
                    self.K = kwargs["K"]
                    self.alpha = kwargs["alpha"]
                    if noise_strategy in ["feature_flipping", "all_flipping"]:
                        if dataset in EMBER_DATASET:
                            self.kbin = KBinsDiscretizer(n_bins=self.K + 1, strategy='uniform', encode='ordinal')
                            self.kbin.fit(self.X)
                            if dataset == "ember_limited":
                                self.limit_id, _, _, _ = load_features(False)
                                self.limit_mask = np.zeros_like(self.X[0])
                                self.limit_mask[self.limit_id] = 1
                        else:
                            assert (self.X >= 0).all() and (self.X <= 1).all()
                    if noise_strategy in ["label_flipping", "all_flipping"]:
                        assert (self.y >= 0).all() and (self.y <= self.K).all()
                elif noise_strategy == "RAB_gaussian":
                    self.sigma = kwargs["sigma"]
                elif noise_strategy == "RAB_uniform":
                    self.a = kwargs["a"]
                    self.b = kwargs["b"]
                else:
                    raise NotImplementedError
            elif dataset in LANGUAGE_DATASET:
                assert noise_strategy in ["sentence_select", "label_flipping", "all_flipping"]
                if noise_strategy in ["sentence_select", "all_flipping"]:
                    self.l = kwargs["l"]
                if noise_strategy in ["label_flipping", "all_flipping"]:
                    self.K = kwargs["K"]
                    self.alpha = kwargs["alpha"]
                    assert (self.y >= 0).all() and (self.y <= self.K).all()
            else:
                raise NotImplementedError

    def process_train(self, key_dict):
        ret_X = self.X.copy()
        ret_y = self.y.copy()  # make sure the original data is not modified
        if self.select_strategy is not None:
            if self.select_strategy in ["bagging_replace", "bagging_wo_replace"]:
                indices = np.random.choice(np.arange(len(self.X)), self.k,
                                           replace=self.select_strategy == "bagging_replace")
                ret_X = ret_X[indices]
                ret_y = ret_y[indices]
            elif self.select_strategy == "binomial":
                pred = np.random.random(len(self.X)) * len(self.X) < self.k
                ret_X = ret_X[pred]
                ret_y = ret_y[pred]

        if self.noise_strategy is not None:
            if self.dataset in FEATURE_DATASET:
                if self.noise_strategy in ["feature_flipping", "all_flipping"]:
                    if self.dataset in EMBER_DATASET:
                        categorized = self.kbin.transform(ret_X) / self.K
                        if self.dataset == "ember_limited":
                            ret_X[:, self.limit_id] = categorized[:, self.limit_id]
                        else:
                            ret_X = categorized

                    mask = np.random.random(ret_X.shape) < self.alpha
                    if self.dataset == "ember_limited":  # protect other features
                        mask *= self.limit_mask
                    delta = np.random.randint(1, self.K + 1, ret_X.shape) / self.K
                    ret_X = ret_X * mask + (1 - mask) * (ret_X + delta)
                    ret_X[ret_X > 1 + 1e-4] -= (1 + self.K) / self.K
                if self.noise_strategy in ["label_flipping", "all_flipping"]:
                    mask = np.random.random(ret_y.shape) < self.alpha
                    delta = np.random.randint(1, self.K + 1, ret_y.shape)
                    ret_y = ret_y * mask + (1 - mask) * (ret_y + delta)
                    ret_y[ret_y > self.K] -= self.K + 1
                if self.noise_strategy == "RAB_gaussian":
                    ret_X += np.random.normal(0, self.sigma, ret_X.shape)
                if self.noise_strategy == "RAB_uniform":
                    ret_X += np.random.uniform(self.a, self.b, ret_X.shape)
            elif self.dataset in LANGUAGE_DATASET:
                if self.noise_strategy in ["sentence_select", "all_flipping"]:
                    maxlen = ret_X.shape[1]
                    ret_X_new = []
                    for x in ret_X:
                        indices = sorted(np.random.choice(np.arange(maxlen), self.l, replace=False))
                        ret_X_new.append(
                            np.pad(x[indices], (0, maxlen - self.l), 'constant', constant_values=(0, 0)))

                    ret_X = np.array(ret_X_new)
                if self.noise_strategy in ["label_flipping", "all_flipping"]:
                    mask = np.random.random(ret_y.shape) < self.alpha
                    delta = np.random.randint(1, self.K + 1, ret_y.shape)
                    ret_y = ret_y * mask + (1 - mask) * (ret_y + delta)
                    ret_y[ret_y > self.K] -= self.K + 1
        if (self.noise_strategy is not None or self.select_strategy is not None) and self.dataset == "imdb":
            for x in ret_X:
                for i in range(len(x)):
                    if x[i] not in key_dict:
                        key_dict[x[i]] = len(key_dict)
                    x[i] = key_dict[x[i]]

        return ret_X, ret_y

    def process_test(self, X, fix_noise):
        ret_X = X.copy()
        if fix_noise:
            if self.noise_strategy is not None:
                if self.dataset in FEATURE_DATASET:
                    if self.noise_strategy in ["feature_flipping", "all_flipping"]:
                        mask = np.random.random(ret_X.shape[1:]) < self.alpha  # fix the noise for each example
                        if self.dataset == "ember_limited":  # protect other features
                            mask *= self.limit_mask
                        delta = np.random.randint(1, self.K + 1, ret_X.shape[1:]) / self.K
                        ret_X = ret_X * mask + (1 - mask) * (ret_X + delta)
                        ret_X[ret_X > 1 + 1e-4] -= (1 + self.K) / self.K
                    if self.noise_strategy == "RAB_gaussian":
                        ret_X += np.random.normal(0, self.sigma, ret_X.shape[1:])  # fix the noise for each example
                    if self.noise_strategy == "RAB_uniform":
                        ret_X += np.random.uniform(self.a, self.b, ret_X.shape[1:])  # fix the noise for each example
                elif self.dataset in LANGUAGE_DATASET:
                    if self.noise_strategy in ["sentence_select", "all_flipping"]:
                        maxlen = ret_X.shape[1]
                        ret_X_new = np.zeros_like(ret_X)
                        indices = sorted(
                            np.random.choice(np.arange(maxlen), self.l,
                                             replace=False))  # fix the noise for each example
                        ret_X_new[:, :self.l] = ret_X[:, indices]
                        ret_X = ret_X_new
        else:
            if self.noise_strategy is not None:
                if self.dataset in FEATURE_DATASET:
                    if self.noise_strategy in ["feature_flipping", "all_flipping"]:
                        mask = np.random.random(ret_X.shape) < self.alpha
                        if self.dataset == "ember_limited":  # protect other features
                            mask *= self.limit_mask
                        delta = np.random.randint(1, self.K + 1, ret_X.shape) / self.K
                        ret_X = ret_X * mask + (1 - mask) * (ret_X + delta)
                        ret_X[ret_X > 1 + 1e-4] -= (1 + self.K) / self.K
                    if self.noise_strategy == "RAB_gaussian":
                        ret_X += np.random.normal(0, self.sigma, ret_X.shape)
                    if self.noise_strategy == "RAB_uniform":
                        ret_X += np.random.uniform(self.a, self.b, ret_X.shape)
                elif self.dataset in LANGUAGE_DATASET:
                    if self.noise_strategy in ["sentence_select", "all_flipping"]:
                        maxlen = ret_X.shape[1]
                        ret_X_new = []
                        for x in ret_X:
                            indices = sorted(np.random.choice(np.arange(maxlen), self.l, replace=False))
                            ret_X_new.append(
                                np.pad(x[indices], (0, maxlen - self.l), 'constant', constant_values=(0, 0)))

                        ret_X = np.array(ret_X_new)

        return ret_X


class DataPreprocessor:
    def __init__(self):
        pass

    @classmethod
    def load(cls, filename, args):
        with open(filename, "rb") as f:
            attack = pickle.load(f)
            this = attack.data_processor
            this.attack = attack
            this.data_processor = DataPreprocessor.build_processor(this.x_train, this.y_train, args)
            return this

    @staticmethod
    def build_processor(x_train, y_train, args):
        return DataProcessor(x_train, y_train, select_strategy=args.select_strategy, k=args.k,
                             noise_strategy=args.noise_strategy, K=args.K, alpha=args.alpha,
                             sigma=args.sigma, a=args.a, b=args.b, dataset=args.dataset, l=args.l)


class MNIST17DataPreprocessor(DataPreprocessor):
    def __init__(self, args):
        super(MNIST17DataPreprocessor, self).__init__()
        # input image dimensions
        img_rows, img_cols = 28, 28

        self.n_classes = 2
        self.n_features = (img_rows, img_cols, 1)

        (x_train, self.y_train), (x_test, self.y_test) = mnist.load_data()
        x_train = x_train[(self.y_train == 1) | (self.y_train == 7)]
        self.y_train = self.y_train[(self.y_train == 1) | (self.y_train == 7)]
        self.y_train = self.y_train > 1
        x_test = x_test[(self.y_test == 1) | (self.y_test == 7)]
        self.y_test = self.y_test[(self.y_test == 1) | (self.y_test == 7)]
        self.y_test = self.y_test > 1

        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        self.x_train = x_train / 255
        self.x_test = x_test / 255
        if args.K != 1:
            raise NotImplementedError("K != 1 not implemented for MNIST17DataPreprocessor.")
        if args.noise_strategy in ["feature_flipping", "all_flipping"]:
            self.x_train = self.x_train >= 0.5
            self.x_test = self.x_test >= 0.5
        if args.noise_strategy in ["label_flipping", "all_flipping"]:
            pass

        self.data_processor = self.build_processor(self.x_train, self.y_train, args)
        print('x_train shape:', x_train.shape, self.y_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')


class MNISTDataPreprocessor(DataPreprocessor):
    def __init__(self, args):
        super(MNISTDataPreprocessor, self).__init__()
        # input image dimensions
        img_rows, img_cols = 28, 28

        self.n_classes = 10
        self.n_features = (img_rows, img_cols, 1)

        (x_train, self.y_train), (x_test, self.y_test) = mnist.load_data()

        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        self.x_train = x_train / 255
        self.x_test = x_test / 255
        if args.noise_strategy in ["label_flipping", "all_flipping"]:
            assert args.K == 9
        if args.noise_strategy in ["feature_flipping", "all_flipping"]:
            self.x_train = np.minimum(np.floor(self.x_train * (args.K + 1)) / args.K, 1)
            self.x_test = np.minimum(np.floor(self.x_test * (args.K + 1)) / args.K, 1)
        if args.noise_strategy in ["label_flipping", "all_flipping"]:
            pass

        self.data_processor = self.build_processor(self.x_train, self.y_train, args)
        print('x_train shape:', x_train.shape, self.y_train.shape)
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')


class IMDBDataPreprocessor(DataPreprocessor):
    def __init__(self, args):
        super(IMDBDataPreprocessor, self).__init__()
        vocab_size = 10000  # Only consider the top 20k words
        self.n_features = args.L  # Only consider the first 200 words of each movie review
        self.n_classes = 2
        (x_train, self.y_train), (x_test, self.y_test) = imdb.load_data(num_words=vocab_size)
        self.x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=self.n_features)
        self.x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=self.n_features)

        self.data_processor = self.build_processor(self.x_train, self.y_train, args)
        print('x_train shape:', self.x_train.shape, self.y_train.shape)
        print(self.x_train.shape[0], 'train samples')
        print(self.x_test.shape[0], 'test samples')


class EmberDataPreProcessor(DataPreprocessor):
    def __init__(self, args):
        super(EmberDataPreProcessor, self).__init__()
        try:
            x_train, y_train, x_test, y_test = ember.read_vectorized_features(
                args.ember_data_dir,
                feature_version=1
            )

        except:
            ember.create_vectorized_features(
                args.ember_data_dir,
                feature_version=1
            )
            x_train, y_train, x_test, y_test = ember.read_vectorized_features(
                args.ember_data_dir,
                feature_version=1
            )

        x_train = x_train.astype(dtype='float64')
        x_test = x_test.astype(dtype='float64')
        if args.K != 1:
            raise NotImplementedError("K != 1 not implemented for EmberDataPreProcessor.")
        # Get rid of unknown labels
        self.x_train = x_train[y_train != -1]
        self.y_train = y_train[y_train != -1]
        self.x_test = x_test[y_test != -1]
        self.y_test = y_test[y_test != -1]
        normal = StandardScaler()
        self.x_train = normal.fit_transform(self.x_train)
        self.x_test = normal.transform(self.x_test)

        self.n_features = x_train.shape[1]
        self.n_classes = 2

        self.data_processor = self.build_processor(self.x_train, self.y_train, args)
        print('x_train shape:', self.x_train.shape, self.y_train.shape)
        print(self.x_train.shape[0], 'train samples')
        print(self.x_test.shape[0], 'test samples')
