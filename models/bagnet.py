#################################################################################################################
# from https://github.com/inspire-group/PatchGuard/blob/master/nets/bagnet.py                                   #
# Adapted from https://github.com/wielandbrendel/bag-of-local-features-models/blob/master/bagnets/pytorchnet.py #
# Mainly changed the model forward() function                                                                   #
#################################################################################################################


import torch.nn as nn
import math
import torch
from torch.utils import model_zoo
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import numpy as np
from tqdm import tqdm
from math import ceil
from torchvision import transforms

# dir_path = os.path.dirname(os.path.realpath(__file__))
#
# __all__ = ['bagnet9', 'bagnet17', 'bagnet33']

model_urls = {
    'bagnet9': 'https://bitbucket.org/wielandbrendel/bag-of-feature-pretrained-models/raw/249e8fa82c0913623a807d9d35eeab9da7dcc2a8/bagnet8-34f4ccd2.pth.tar',
    'bagnet17': 'https://bitbucket.org/wielandbrendel/bag-of-feature-pretrained-models/raw/249e8fa82c0913623a807d9d35eeab9da7dcc2a8/bagnet16-105524de.pth.tar',
    'bagnet33': 'https://bitbucket.org/wielandbrendel/bag-of-feature-pretrained-models/raw/249e8fa82c0913623a807d9d35eeab9da7dcc2a8/bagnet32-2ddd53ed.pth.tar',
}

from models.Model import Model
from utils.defense_utils import pg2_detection_provable, pg2_detection


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, kernel_size=1):
        super(Bottleneck, self).__init__()
        # #print('Creating bottleneck with kernel size {} and stride {} with padding {}'.format(kernel_size, stride, (kernel_size - 1) // 2))
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=stride,
                               padding=0, bias=False)  # changed padding from (kernel_size - 1) // 2
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, **kwargs):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if residual.size(-1) != out.size(-1):
            diff = residual.size(-1) - out.size(-1)
            residual = residual[:, :, :-diff, :-diff]

        out += residual
        out = self.relu(out)

        return out


class BagNet(nn.Module):

    def __init__(self, block, layers, strides=[1, 2, 2, 2], kernel3=[0, 0, 0, 0], num_classes=1000, clip_range=None,
                 aggregation='mean', in_channels=3):
        self.inplanes = 64
        super(BagNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=1, stride=1, padding=0,
                               bias=False)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.001)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], kernel3=kernel3[0], prefix='layer1')
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], kernel3=kernel3[1], prefix='layer2')
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], kernel3=kernel3[2], prefix='layer3')
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], kernel3=kernel3[3], prefix='layer4')
        self.avgpool = nn.AvgPool2d(1, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.block = block

        self.clip_range = clip_range
        self.aggregation = aggregation

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, kernel3=0, prefix=''):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        kernel = 1 if kernel3 == 0 else 3

        layers.append(block(self.inplanes, planes, stride, downsample, kernel_size=kernel))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            kernel = 1 if kernel3 <= i else 3

            layers.append(block(self.inplanes, planes, kernel_size=kernel))

        return nn.Sequential(*layers)

    def forward(self, x, y=None):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = x.permute(0, 2, 3, 1)

        x = self.fc(x)
        if self.clip_range is not None:
            x = torch.clamp(x, self.clip_range[0], self.clip_range[1])
        if self.aggregation == 'mean':
            x = torch.mean(x, dim=(1, 2))
        elif self.aggregation == 'median':
            x = x.view([x.size()[0], -1, 10])
            x = torch.median(x, dim=1)
            return x.values
        elif self.aggregation == 'cbn':  # clipped BagNet
            x = torch.tanh(x * 0.05 - 1)
            x = torch.mean(x, dim=(1, 2))
        elif self.aggregation == 'adv':  # provable adversarial training
            window_size = 1  # the size of window to be masked during the training
            B, W, H, C = x.size()
            x = torch.clamp(x, 0, torch.tensor(float('inf')))  # clip
            tmp = x[torch.arange(B), :, :, y]  # the feature map for the true class
            tmp = tmp.unfold(1, window_size, 1).unfold(2, window_size, 1)  # unfold
            tmp = tmp.reshape([B, -1, window_size, window_size])  # [B,num_window,window_size,window_size]
            tmp = torch.sum(tmp, axis=(-2, -1))  # [B,num_window] true class evidence in every window
            tmp = torch.max(tmp, axis=-1).values  # [B] max window class evidence
            x = torch.sum(x, dim=(1, 2))  #
            x[torch.arange(B), y] -= tmp  # substract the max true window class evidence
            x /= (W * H)
        elif self.aggregation == 'none':
            pass

        return x


def bagnet33(pretrained=False, strides=[2, 2, 2, 1], **kwargs):
    """Constructs a Bagnet-33 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = BagNet(Bottleneck, [3, 4, 6, 3], strides=strides, kernel3=[1, 1, 1, 1], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['bagnet33']))
    return model


def bagnet17(pretrained=False, strides=[2, 2, 2, 1], **kwargs):
    """Constructs a Bagnet-17 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = BagNet(Bottleneck, [3, 4, 6, 3], strides=strides, kernel3=[1, 1, 1, 0], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['bagnet17']))
    return model


def bagnet9(pretrained=False, strides=[2, 2, 2, 1], **kwargs):
    """Constructs a Bagnet-9 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = BagNet(Bottleneck, [3, 4, 6, 3], strides=strides, kernel3=[1, 1, 0, 0], **kwargs)
    # model = BagNet(Bottleneck, [2,2,2,2], strides=strides, kernel3=[1,1,0,0], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['bagnet9']))
    return model


def bagnet5(pretrained=False, strides=[2, 2, 1, 1], **kwargs):
    if not pretrained:
        return BagNet(Bottleneck, [3, 4, 6, 3], strides=strides, kernel3=[1, 0, 0, 0], **kwargs)
    else:
        model = BagNet(Bottleneck, [3, 4, 6, 3], strides=strides, kernel3=[1, 0, 0, 0], **kwargs)
        model_1 = bagnet9(pretrained=True, clip_range=None, aggregation='mean')
        if kwargs["in_channels"] == 1:
            model.conv1 = nn.Conv2d(1, 64, kernel_size=1, stride=1, padding=0,
                                    bias=False)
        else:
            model.conv1 = model_1.conv1
        model.conv2 = model_1.conv2
        model.bn1 = model_1.bn1
        model.relu = model_1.relu
        model.layer1 = model_1.layer1
        model.layer4 = model_1.layer4
        model.fc = nn.Linear(model.fc.in_features, kwargs["num_classes"])
        return model


class BagNetModel(Model):
    def __init__(self, input_shape, n_classes, lr, device, patch_size, weight_decay, x_test, y_test, wandb, tau=0.5,
                 pretrained=False):
        self.pretrained = pretrained
        super().__init__(input_shape, n_classes, lr)
        self.device = device
        self.tau = tau
        self.patch_size = patch_size
        self.rf_size = 5
        self.weight_decay = weight_decay
        self.x_test = x_test
        self.y_test = y_test
        self.wandb = wandb
        self.test_res = {}

    def build_model(self):
        # if self.pretrained:
        #     model = bagnet9(pretrained=True, clip_range=None, aggregation='mean')
        #     model.fc = nn.Linear(model.fc.in_features, self.n_classes)
        #     if self.input_shape[-1] == 1:
        #         model.conv1 = nn.Conv2d(1, 64, kernel_size=1, stride=1, padding=0,
        #                        bias=False)
        #     return model
        # else:
        #     return bagnet9(pretrained=False, clip_range=None, aggregation='mean', num_classes=self.n_classes,
        #                    in_channels=self.input_shape[-1])
        return bagnet5(pretrained=self.pretrained, aggregation='mean', num_classes=self.n_classes,
                       in_channels=self.input_shape[-1])

    def save(self, save_path, file_name="0", predictions=None):
        torch.save(self.model.state_dict(), os.path.join(save_path, file_name))
        if predictions is not None:
            np.save(os.path.join(save_path, file_name + "_predictions"), predictions)

    def load(self, save_path, file_name):
        self.model.load_state_dict(torch.load(os.path.join(save_path, file_name)))

    def data_aug(self, data, **kwargs):
        data = transforms.RandomCrop(data.shape[-1], 3)(data)
        data = transforms.RandomRotation(10)(data)
        return data

    def fit(self, X, y, batch_size, epochs, x_test=None, y_test=None):
        X = np.transpose(X, (0, 3, 1, 2))
        self.mean = np.expand_dims(np.mean(X, axis=(0, 2, 3), dtype='float64'), axis=(1, 2))
        self.std = np.expand_dims(np.std(X, axis=(0, 2, 3), dtype='float64'), axis=(1, 2))
        X = (X - self.mean) / self.std
        print(self.mean, self.std)
        data = TensorDataset(torch.Tensor(X), torch.Tensor(y).long().max(dim=-1)[1])
        loader = DataLoader(data, batch_size=batch_size, shuffle=True, pin_memory=True)

        if self.x_test is not None:
            x_test = np.transpose(self.x_test, (0, 3, 1, 2))
            x_test = (x_test - self.mean) / self.std
            data = TensorDataset(torch.Tensor(x_test),
                                 torch.Tensor(self.y_test).long())
            batch_size = 256
            testloader = DataLoader(data, batch_size=batch_size, shuffle=False, pin_memory=True)
        else:
            testloader = None

        self.model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        for epoch in range(epochs):
            self.model.train()
            self.model.aggregation = 'mean'
            train_loss = 0
            correct = 0
            total = 0
            wandb_log = {}
            for batch_idx, (inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                inputs = self.data_aug(inputs)
                optimizer.zero_grad()
                outputs = self.model(inputs, y=targets)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            # if (epoch + 1) % 1 == 0:
            #     print(f"Train Accuracy: {round(100. * correct / total, 2)}, "
            #           f"Loss: {round(train_loss / len(loader), 4)} at epoch {epoch}")
            #     wandb_log["train_loss"] = train_loss / len(loader)
            #     wandb_log["train_acc"] = 100. * correct / total
            # if (epoch + 1) % 1 == 0:
            #     if testloader is not None:
            #         self.model.eval()
            #         self.model.aggregation = 'mean'
            #         correct = 0
            #         total = 0
            #         test_loss = 0
            #         for batch_idx, (inputs, targets) in enumerate(testloader):
            #             inputs, targets = inputs.to(self.device), targets.to(self.device)
            #             outputs = self.model(inputs, y=targets)
            #             loss = criterion(outputs, targets)
            #             test_loss += loss.item()
            #             _, predicted = outputs.max(1)
            #             total += targets.size(0)
            #             correct += predicted.eq(targets).sum().item()
            #
            #         print(f"Test Accuracy: {round(100. * correct / total, 2)}, "
            #               f"Loss: {round(test_loss / len(testloader), 4)} at epoch {epoch}")
            #         wandb_log["test_loss"] = test_loss / len(testloader)
            #         wandb_log["test_acc"] = 100. * correct / total
            #         if epoch not in self.test_res:
            #             self.test_res[epoch] = []
            #         self.test_res[epoch].append(round(100. * correct / total, 2))

            if self.wandb is not None:
                self.wandb.log(wandb_log, commit=True)

        # for epoch in sorted(self.test_res.keys()):
        #     print(epoch, np.mean(self.test_res[epoch]))

    def evaluate(self, x_test, y_test, tune=True):
        x_test = np.transpose(x_test, (0, 3, 1, 2))
        x_test = (x_test - self.mean) / self.std
        data = TensorDataset(torch.Tensor(x_test),
                             torch.Tensor(y_test).long().max(dim=-1)[1])
        batch_size = 256
        loader = DataLoader(data, batch_size=batch_size, shuffle=False, pin_memory=True)
        verified = []
        result_list = []
        confs = []
        clean_corr = 0
        rf_stride = 4
        window_size = ceil((self.patch_size + self.rf_size - 1) / rf_stride)
        self.model.eval()
        self.model.aggregation = 'none'
        with torch.no_grad():
            for data, labels in tqdm(loader):
                data = data.to(self.device)
                labels = labels.numpy()
                output_clean = self.model(data).detach().cpu().numpy()  # logits
                # output_clean = softmax(output_clean,axis=-1) # confidence
                # output_clean = (output_clean > 0.2).astype(float) # predictions with confidence threshold

                # note: the provable analysis of robust masking is cpu-intensive and can take some time to finish
                # you can dump the local feature and do the provable analysis with another script so that GPU mempry is not always occupied
                for i in range(len(labels)):
                    local_feature = output_clean[i]
                    # result,clean_pred = provable_detection(local_feature,labels[i],tau=args.tau,window_shape=[window_size,window_size])
                    # clean_corr += clean_pred

                    pred, vri, conf = pg2_detection(local_feature, tau=self.tau,
                                                    window_shape=[window_size, window_size])
                    # clean_corr += clean_pred == labels[i]

                    # result = pg2_detection_provable(local_feature, labels[i], tau=self.tau,
                    #                                 window_shape=[window_size, window_size])
                    result_list.append(pred)
                    verified.append(vri)
                    confs.append(conf)

                # acc_clean = np.sum(np.argmax(np.mean(output_clean, axis=(1, 2)), axis=1) == labels)
                # accuracy_list.append(acc_clean)

        predictions = np.array(result_list)
        # verified = np.array(verified)
        confs = np.array(confs)
        print("Test accuracy: ", np.mean(predictions == np.argmax(y_test, axis=-1)))
        if tune:
            self.verify(predictions, confs, y_test)
        return predictions, confs

    def verify(self, predictions, confs, y_test):
        for tau in np.linspace(0.1, 1, 10):
            print("Tau: ", tau)
            verified = confs <= tau
            predictions_cert = predictions * verified + (1 - verified) * self.n_classes
            certified_cor = np.mean(predictions_cert == np.argmax(y_test, axis=-1))
            print("Certified accuracy: ", round(certified_cor * 100, 3))
            incorrect = np.mean((predictions_cert != np.argmax(y_test, axis=-1)) * verified)
            print("Incorrectly certified: ", round(incorrect * 100, 3))
            print("Abstained: ", round((1 - certified_cor - incorrect) * 100, 3))
            # print("Approximate bd accuracy: ",
            #       np.mean(predictions_cert == np.argmax(y_test, axis=-1)) - incorrect / (self.n_classes - 1))
        # print("Mean confidence: ", np.mean(confs))
