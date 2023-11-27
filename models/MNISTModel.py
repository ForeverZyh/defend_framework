from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D
from tensorflow.keras.regularizers import l2
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import os
import torch.optim as optim
from torchvision import transforms

from models.Model import Model


class MNISTModel(Model):
    def __init__(self, n_features, n_classes, lr=1e-3, weight_decay=1e-2):
        self.weight_decay = weight_decay
        super(MNISTModel, self).__init__(n_features, n_classes, lr)

    # def build_model(self):
    #     def scheduler(epoch, lr):
    #         if epoch > 0 and epoch % 100 == 0:
    #             lr *= 0.5
    #         # print(lr)
    #         return lr
    #
    #     reg = l2(1e-3)
    #     model = Sequential()
    #     model.add(Conv2D(16, kernel_size=(5, 5),
    #                      activation='relu',
    #                      input_shape=self.input_shape))
    #     model.add(AveragePooling2D(pool_size=(2, 2)))
    #     model.add(Conv2D(32, (5, 5), activation='relu'))
    #     model.add(AveragePooling2D(pool_size=(2, 2)))
    #     model.add(Flatten())
    #     model.add(Dropout(0.25))
    #     model.add(Dense(32, activation='relu', kernel_regularizer=reg, bias_regularizer=reg))
    #     model.add(Dropout(0.25))
    #     model.add(Dense(512, activation='relu', kernel_regularizer=reg, bias_regularizer=reg))
    #     model.add(Dense(self.n_classes, activation='softmax'))
    #
    #     model.compile(loss=keras.losses.categorical_crossentropy,
    #                   optimizer=keras.optimizers.Adam(lr=self.lr),
    #                   metrics=['accuracy'])
    #     # self.callback = [keras.callbacks.LearningRateScheduler(scheduler)]
    #     return model

    def build_model(self):
        model = Sequential()
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.n_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.AdamW(learning_rate=self.lr, weight_decay=self.weight_decay),
                      metrics=['accuracy'])
        # self.callback = [keras.callbacks.LearningRateScheduler(scheduler)]
        return model


class mlp_3layer(nn.Module):
    def __init__(self, in_ch, in_dim, width=1):
        super(mlp_3layer, self).__init__()
        self.fc1 = nn.Linear(in_ch * in_dim * in_dim, 256 * width)
        self.fc2 = nn.Linear(256 * width, 128 * width)
        self.fc3 = nn.Linear(128 * width, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MNISTTorchModel(Model):
    def __init__(self, input_shape, n_classes, lr, device, weight_decay, data_processor):
        super().__init__(input_shape, n_classes, lr)
        self.device = device
        self.weight_decay = weight_decay
        self.test_res = {}
        self.data_processor = data_processor

    def build_model(self):
        return mlp_3layer(self.input_shape[-1], self.input_shape[0])

    def save(self, save_path, file_name="0", predictions=None):
        torch.save(self.model.state_dict(), os.path.join(save_path, file_name))
        if predictions is not None:
            np.save(os.path.join(save_path, file_name + "_predictions"), predictions)

    def load(self, save_path, file_name):
        self.model.load_state_dict(torch.load(os.path.join(save_path, file_name)))

    def data_aug(self, data, **kwargs):
        data_aug = transforms.RandomCrop(data.shape[-1], 3)(data)
        data_aug = transforms.RandomRotation(10)(data_aug)
        data_np = self.data_processor.noise_data(data)
        return torch.cat([data, data_aug, data_np], dim=0)

    def fit(self, X, y, batch_size, epochs, x_test=None, y_test=None):
        X = np.transpose(X, (0, 3, 1, 2))
        data = TensorDataset(torch.Tensor(X), torch.Tensor(y).long().max(dim=-1)[1])
        loader = DataLoader(data, batch_size=batch_size, shuffle=True, pin_memory=True)

        if x_test is not None:
            x_test = np.transpose(x_test, (0, 3, 1, 2))
            data = TensorDataset(torch.Tensor(x_test),
                                 torch.Tensor(y_test).long().max(dim=-1)[1])
            batch_size = 256
            testloader = DataLoader(data, batch_size=batch_size, shuffle=False, pin_memory=True)
        else:
            testloader = None

        self.model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            correct = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs = self.data_aug(inputs)
                targets = torch.cat([targets, targets, targets], dim=0)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            continue
            if (epoch + 1) % 1 == 0:
                print(f"Train Accuracy: {round(100. * correct / total, 2)}, "
                      f"Loss: {round(train_loss / len(loader), 4)} at epoch {epoch}")
            if (epoch + 1) % 1 == 0:
                if testloader is not None:
                    self.model.eval()
                    correct = 0
                    total = 0
                    test_loss = 0
                    for batch_idx, (inputs, targets) in enumerate(testloader):
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        outputs = self.model(inputs)
                        loss = criterion(outputs, targets)
                        test_loss += loss.item()
                        _, predicted = outputs.max(1)
                        total += targets.size(0)
                        correct += predicted.eq(targets).sum().item()

                    print(f"Test Accuracy: {round(100. * correct / total, 2)}, "
                          f"Loss: {round(test_loss / len(testloader), 4)} at epoch {epoch}")
                    if epoch not in self.test_res:
                        self.test_res[epoch] = []
                    self.test_res[epoch].append(round(100. * correct / total, 2))

        for epoch in sorted(self.test_res.keys()):
            print(epoch, np.mean(self.test_res[epoch]))

    def evaluate(self, x_test, y_test):
        x_test = np.transpose(x_test, (0, 3, 1, 2))
        data = TensorDataset(torch.Tensor(x_test),
                             torch.Tensor(y_test).long().max(dim=-1)[1])
        batch_size = 256
        testloader = DataLoader(data, batch_size=batch_size, shuffle=False, pin_memory=True)

        self.model.eval()
        correct = 0
        total = 0
        prediction_label = []
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            prediction_label.extend(list(predicted.cpu().numpy()))

        print(f"Test Accuracy: {round(100. * correct / total, 2)}")
        return np.array(prediction_label).astype(np.int32)


class MNIST01Model(Model):
    def __init__(self, n_features, n_classes, lr=1e-3):
        super(MNIST01Model, self).__init__(n_features, n_classes, lr)

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(16, kernel_size=(5, 5),
                         activation='relu',
                         input_shape=self.input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        model.add(Conv2D(32, (5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.n_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(lr=self.lr),
                      metrics=['accuracy'])
        return model


class MNIST17Model(Model):
    def __init__(self, n_features, n_classes, lr=1e-3):
        super(MNIST17Model, self).__init__(n_features, n_classes, lr)

    def build_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.input_shape))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.n_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(lr=self.lr),
                      metrics=['accuracy'])
        return model


class FMNISTModel(Model):
    def __init__(self, n_features, n_classes, lr=1e-3):
        super(FMNISTModel, self).__init__(n_features, n_classes, lr)

    def build_model(self):
        reg = l2(1e-3)
        model = Sequential()
        model.add(Conv2D(16, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=self.input_shape))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        model.add(Flatten())
        model.add(Dropout(0.6))
        model.add(Dense(128, activation='relu', kernel_regularizer=reg, bias_regularizer=reg))
        model.add(Dropout(0.5))
        model.add(Dense(self.n_classes, activation='softmax'))

        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adam(lr=self.lr),
                      metrics=['accuracy'])
        return model
