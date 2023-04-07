import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
from torch.nn import CrossEntropyLoss
from auto_LiRPA import BoundedTensor, PerturbationLpNorm

from models.LiRPAModel import LiRPAModel


class EmberModel(LiRPAModel):
    def __init__(self, n_features, n_classes, args, device, lr=1e-3):
        super(EmberModel, self).__init__([1, 1, n_features], n_classes, args, device, mlp_4layer, lr)

    def data_aug(self, data, **kwargs):
        data += torch.clamp(torch.normal(torch.zeros_like(data), torch.ones_like(data) * 0.1), min=0, max=1)
        return data

    def adv_attack_l0_one_pixel(self, data, target, eps: float):
        data = self.data_aug(data)

        def clip_delta(d, eps):
            # retain the element with the largest absolute value, setting other to zero
            d = torch.clamp(d, min=-eps, max=eps)
            d = d.reshape(d.shape[0], -1)
            _, idx = d.abs().max(dim=1, keepdim=True)
            d = torch.zeros_like(d).scatter_(1, idx, d.gather(1, idx))
            mask = torch.zeros_like(d).scatter_(1, idx, torch.ones_like(idx, dtype=torch.float32))
            mask = mask.reshape(data.shape)
            d = d.reshape(data.shape)
            return d, mask

        # adapted from torchattack
        adv_images = data + torch.empty_like(data).uniform_(-eps, eps)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        delta, _ = clip_delta(adv_images - data, eps)
        adv_images = torch.clamp(data + delta, min=0, max=1).detach()

        alpha = self.args.SABR_alpha
        for i in range(self.args.SABR_step):
            if i == 4 or i == 7:
                alpha *= 0.1
            adv_images.requires_grad = True
            output = self.model(adv_images)
            regular_ce = CrossEntropyLoss()(output, target)  # regular CrossEntropyLoss used for warming up

            # Update adversarial images
            grad = torch.autograd.grad(regular_ce, adv_images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + alpha * grad.sign()
            delta, _ = clip_delta(adv_images - data, eps)
            adv_images = torch.clamp(data + delta, min=0, max=1).detach()

        delta, mask = clip_delta(adv_images - data, eps * (1 - self.args.SABR_lambda))
        adv_images = torch.clamp(data + delta, min=0, max=1).detach()

        return adv_images, mask.detach()

    def fit(self, X, y, batch_size, epochs, dummy=None):
        X = np.expand_dims(X, axis=[1, 2])

        if not self.args.SABR:
            aug_func = self.data_aug
        else:
            aug_func = self.adv_attack_l0_one_pixel

        super(EmberModel, self).fit(X, y, batch_size, epochs, aug_func)

    def evaluate(self, x_test, y_test):
        x_test = np.expand_dims(x_test, axis=[1, 2])
        return super(EmberModel, self).evaluate(x_test, y_test)

    def test_exhaustive(self, batch_size, loader):
        self.model.eval()
        predictions = np.array([], dtype=int)
        verified = np.array([], dtype=bool)
        verified_cnt = 0

        with tqdm.tqdm(total=len(loader.dataset)) as pbar:
            for i, (data, _) in enumerate(loader):
                if list(self.model.parameters())[0].is_cuda:
                    data = data.cuda()
                output = self.model(data)
                batch_predictions = torch.argmax(output, dim=1)
                batch_predictions = batch_predictions.cpu().numpy()
                predictions = np.append(predictions, batch_predictions)
                assert data.shape[1] == 1 and data.shape[2] == 1  # only the last dim is the feature dim
                batch_vf = []
                # print("batch ", i)
                for data_idx in range(len(data)):
                    # print("=" * 10, "data_idx ", data_idx, "=" * 10)
                    remain = [(0, data.shape[-1])]
                    cur_remain = data.shape[-1]
                    adaptive_length = 1
                    while len(remain) > 0:
                        # print("remain percentage: ", cur_remain / data.shape[-1] * 100, "%")
                        data_ = data[data_idx].repeat(batch_size, 1, 1, 1)
                        data_lb = data[data_idx].repeat(batch_size, 1, 1, 1)
                        data_ub = data[data_idx].repeat(batch_size, 1, 1, 1)
                        idx_range = [None] * batch_size
                        for j in range(batch_size):
                            if len(remain) == 0:
                                break
                            start, length = remain.pop()
                            if length <= adaptive_length:
                                idx_range[j] = (start, length)
                                data_lb[j, :, :, start:start + length] = 0
                                data_ub[j, :, :, start:start + length] = 1
                                cur_remain -= length
                            else:
                                remain.append((start + adaptive_length, length - adaptive_length))
                                idx_range[j] = (start, adaptive_length)
                                data_lb[j, :, :, start:start + adaptive_length] = 0
                                data_ub[j, :, :, start:start + adaptive_length] = 1
                                cur_remain -= adaptive_length

                        ptb = PerturbationLpNorm(norm=np.inf, eps=1, x_L=data_lb, x_U=data_ub)
                        x = BoundedTensor(data_, ptb)
                        output = self.model(x)
                        batch_predictions = torch.argmax(output, dim=1)
                        # generate specifications
                        c = torch.eye(self.n_classes).type_as(data_)[batch_predictions].unsqueeze(1) - torch.eye(
                            self.n_classes).type_as(
                            data_).unsqueeze(0).to(data_.device)
                        # remove specifications to self
                        I = (~(batch_predictions.data.unsqueeze(1) == torch.arange(self.n_classes).type_as(
                            batch_predictions.data).unsqueeze(0))).to(data_.device)
                        c = (c[I].view(data_.size(0), self.n_classes - 1, self.n_classes))
                        lb, ub = self.model.compute_bounds(IBP=True, C=c, method=None)
                        lb, ub = self.model.compute_bounds(IBP=False, C=c, method="backward", bound_upper=False)
                        batch_verified = (lb > 0).all(dim=1).cpu().numpy()
                        # print(batch_verified)
                        fail = False
                        for j in range(batch_size):
                            if idx_range[j] is None:
                                continue
                            if not batch_verified[j]:
                                cur_remain += idx_range[j][1]
                                remain.append(idx_range[j])
                                if idx_range[j][1] == 1:
                                    fail = True
                                    break
                        if fail:
                            # print("failed at ", remain[-1][0])
                            batch_vf.append(False)
                            break
                        if not batch_verified.all():
                            adaptive_length = adaptive_length // 2
                            # print("partial verified, reduce length to", adaptive_length)
                        else:
                            adaptive_length = adaptive_length * 2
                            # print("verified, increase length to", adaptive_length)
                    if len(remain) == 0:
                        # print("verified")
                        batch_vf.append(True)
                        verified_cnt += 1
                    pbar.update(1)
                    pbar.set_postfix({"verified": verified_cnt / (len(verified) + len(batch_vf)) * 100})
                verified = np.append(verified, batch_vf)

        return predictions, verified


class mlp_4layer(nn.Module):
    def __init__(self, in_ch, in_dim):
        super(mlp_4layer, self).__init__()
        self.fc1 = nn.Linear(in_ch * in_dim, 2000)
        self.fc2 = nn.Linear(2000, 1000)
        self.fc3 = nn.Linear(1000, 100)
        self.fc4 = nn.Linear(100, 2)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
