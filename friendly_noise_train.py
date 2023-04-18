"""
https://github.com/tianyu139/friendly-noise/blob/master/main.py
"""

import argparse
import logging
import os
import random
import shutil
import time
import datetime
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torchvision
import pickle
from torchvision import datasets
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import copy
import pandas as pd

from friendly_noise.friendly_noise import generate_friendly_noise, UniformNoise, GaussianNoise, BernoulliNoise
from utils.data_processing import get_ember


class mlp_4layer(nn.Module):
    def __init__(self, in_ch, in_dim):
        super(mlp_4layer, self).__init__()
        self.fc1 = nn.Linear(in_ch * in_dim, 2000)
        self.bn1 = nn.BatchNorm1d(2000)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(2000, 1000)
        self.bn2 = nn.BatchNorm1d(1000)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(1000, 100)
        self.bn3 = nn.BatchNorm1d(100)
        self.dropout3 = nn.Dropout(0.5)
        self.fc4 = nn.Linear(100, 2)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.bn3(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        return x


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MAX_VALUE = 100
best_acc = 0


class EmberPoisonDataPreProcessor():
    def __init__(self, args):
        with open(os.path.join(args.load_poison_dir, "watermarked_train"), "rb") as f:
            is_d, poisoned_X, poisoned_id = pickle.load(f)
            print("discretize:", is_d)

        self.x_train, self.y_train, x_test, y_test = get_ember(args, is_d)
        mw_train = self.x_train[self.y_train == 1]
        gw_train = self.x_train[self.y_train == 0]
        # TODO add arg poisoned_rate
        if len(poisoned_id) > 600:
            print(f"load {len(poisoned_id)} poisoned training examples... Truncated to 600!")
            poisoned_id = poisoned_id[:600]
            poisoned_X = poisoned_X[:600]

        gw_train[poisoned_id] = poisoned_X
        self.x_train = np.vstack([gw_train, mw_train])
        self.y_train = np.array([0] * len(gw_train) + [1] * len(mw_train))
        self.x_test = np.load(os.path.join(args.load_poison_dir, "watermarked_X_test.npy"))
        self.y_test = np.load(os.path.join(args.load_poison_dir, "watermarked_y_test.npy"))
        x_test = x_test[y_test == 1]
        y_test = y_test[y_test == 1]
        self.x_test = np.concatenate((self.x_test, x_test), axis=0)
        self.y_test = np.concatenate((self.y_test, y_test), axis=0)

        self.n_features = self.x_train.shape[1]
        self.n_classes = 2

        print('x_train shape:', self.x_train.shape, self.y_train.shape)
        print(self.x_train.shape[0], 'train samples')
        print(self.x_test.shape[0], 'test samples')


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class PerturbedPoisonedDataset():
    '''
    trainset - full training set containing all indices
    indices  - subset of training set
    poison_instances - list of tuples of poison examples
                       and their respective labels like
                       [(x_0, y_0), (x_1, y_1) ...]
                       this must correspond to poison indices
    poison_indices - list of indices which are poisoned
    transform - transformation to apply to each image
    return_index - whether to return original index into trainset
    perturbations - perturbations before normalization
    '''

    def __init__(self, trainset, indices, poison_instances=[], poison_indices=[], transform=None, return_index=False,
                 size=None, perturbations=None, convert_tensor=True):
        if len(poison_indices) or len(poison_instances):
            assert len(poison_indices) == len(poison_instances)
        self.trainset = trainset
        self.indices = indices
        self.transform = transform
        self.poisoned_label = (
            None if len(poison_instances) == 0 else poison_instances[0][1]
        )
        self.return_index = return_index
        if size is None:
            size = len(indices)
        if size < len(indices):
            self.find_indices(size, poison_indices, poison_instances)

        # Set up new indexing
        if len(poison_instances) > 0:

            poison_mask = np.isin(poison_indices, self.indices)
            poison_mask = np.nonzero(poison_mask)[0]
            self.poison_map = {
                int(poison_indices[i]): poison_instances[i]
                for i in poison_mask
            }

            clean_mask = np.isin(self.indices, poison_indices, invert=True)
            self.clean_indices = self.indices[clean_mask]
        else:
            self.poison_map = {}
            self.clean_indices = self.indices

        if perturbations is None:
            self.perturbations = [None for x in indices]
        else:
            self.perturbations = perturbations

        self.to_pil = transforms.ToPILImage()
        self.convert_tensor = convert_tensor
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.indices)

    def set_perturbations(self, perturbations):
        if perturbations is None:
            self.perturbations = [None for x in indices]
        else:
            self.perturbations = perturbations

    def __getitem__(self, index):
        if self.perturbations[index] is not None:
            perturb = self.perturbations[index]
        else:
            perturb = None

        index = self.indices[index]
        if index in self.poison_map:
            img, label = self.poison_map[index]
            # p = 1
        else:
            img, label = self.trainset[index]
            # p = 0
        if self.convert_tensor:
            img = self.to_tensor(img)
        if self.transform is not None:
            # try:
            if perturb is not None:
                # print(f"Adding perturbations with mag: {torch.abs(perturb).mean()}")
                img += perturb
                img = torch.clamp(img, 0, 1)

            img = self.transform(img)
        if self.return_index:
            return img, label, index
        else:
            return img, label  # 0 for unpoisoned, 1 for poisoned

    def find_indices(self, size, poison_indices, poison_instances):
        good_idx = np.array([])
        batch_tar = np.array(self.trainset.targets)
        num_classes = len(set(batch_tar))
        num_per_class = int(size / num_classes)
        for label in range(num_classes):
            all_idx_for_this_class = np.where(batch_tar == label)[0]
            all_idx_for_this_class = np.setdiff1d(
                all_idx_for_this_class, poison_indices
            )
            this_class_idx = all_idx_for_this_class[:num_per_class]
            if label == self.poisoned_label and len(poison_instances) > 0:
                num_clean = num_per_class - len(poison_instances)
                this_class_idx = this_class_idx[:num_clean]
                this_class_idx = np.concatenate((this_class_idx, poison_indices))
            good_idx = np.concatenate((good_idx, this_class_idx))
        good_idx = good_idx.astype(int)
        self.indices = good_idx[np.isin(good_idx, self.indices)]


def train(args, trainloader, noaug_trainloader, test_loader, model, optimizer, scheduler, target_img, target_class,
          poisoned_label, train_dataset, loss_fn, poison_indices, base_dataset, poison_tuples):
    global best_acc
    test_bd_mw_accs = []
    test_clean_mw_accs = []
    test_clean_gw_accs = []
    cluster = []
    time_start = time.time()
    end = time.time()

    model.train()
    N = args.train_size
    weights = torch.ones(N)
    times_selected = torch.zeros(N)
    x_test, y_test = test_loader

    for epoch in range(args.start_epoch, args.epochs):
        logger.info(f"***** Epoch {epoch} *****")
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        # num_poison_selected = torch.tensor(0)

        args.eval_step = len(trainloader)
        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step))
        model.train()

        if 'friendly' in args.noise_type and epoch == args.friendly_begin_epoch:
            logger.info(
                f"Generating friendly noise: epochs={args.friendly_epochs}  mu={args.friendly_mu} lr={args.friendly_lr} loss={args.friendly_loss}")
            out = generate_friendly_noise(
                model,
                noaug_trainloader,
                args.device,
                friendly_epochs=args.friendly_epochs,
                mu=args.friendly_mu,
                friendly_lr=args.friendly_lr,
                clamp_min=-args.friendly_clamp / MAX_VALUE,
                clamp_max=args.friendly_clamp / MAX_VALUE,
                return_preds=args.save_friendly_noise,
                loss_fn=args.friendly_loss)
            model.zero_grad()
            model.train()

            if args.save_friendly_noise:
                friendly_noise, original_preds = out
                model_to_save = model.module if hasattr(model, "module") else model
                # save_checkpoint({
                #     'epoch': epoch + 1,
                #     'state_dict': model_to_save.state_dict(),
                #     'acc': test_acc,
                #     'best_acc': best_acc,
                #     'poison_acc': p_acc,
                #     'best_poison_acc': best_p_acc,
                #     'optimizer': optimizer.state_dict(),
                #     'scheduler': scheduler.state_dict(),
                #     'times_selected': times_selected,
                # }, is_best=False, checkpoint=args.out, filename='model-friendlygen.pth.tar')
                friendly_path = os.path.join(args.out, 'friendly-friendlygen.npy')
                pred_path = os.path.join(args.out, 'original-preds-friendlygen.npy')
                np.save(friendly_path, friendly_noise.numpy())
                np.save(pred_path, original_preds.numpy())
                logger.info(f"Saved friendly noise to {friendly_path}")
                logger.info(f"Saved original predictions to {pred_path}")
            else:
                friendly_noise = out

            trainloader.dataset.set_perturbations(friendly_noise)
            logger.info(
                f"Friendly noise stats:  Max: {torch.max(friendly_noise)}  Min: {torch.min(friendly_noise)}  Mean (abs): {torch.mean(torch.abs(friendly_noise))}  Mean: {torch.mean(friendly_noise)}")

        for batch_idx, batch_input in enumerate(trainloader):
            input, targets_u_gt, index = batch_input
            targets_u_gt = targets_u_gt.long()
            # num_poison_selected += torch.sum(p)

            data_time.update(time.time() - end)
            logits_u_w = model(input.to(args.device))
            pseudo_label = torch.softmax(logits_u_w, dim=-1)
            probs_u, targets_u = torch.sort(pseudo_label, dim=-1, descending=True)
            max_probs, targets_u = probs_u[:, 0], targets_u[:, 0]

            loss = loss_fn(logits_u_w, targets_u_gt.to(args.device))
            loss = loss.mean()

            optimizer.zero_grad()
            loss.backward()
            losses.update(loss.item())
            optimizer.step()
            model.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()

            if not args.no_progress:
                p_bar.set_description(
                    "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. ".format(
                        epoch=epoch + 1,
                        epochs=args.epochs,
                        batch=batch_idx + 1,
                        iter=args.eval_step,
                        lr=scheduler.get_last_lr()[0],
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg))
                p_bar.update()

            times_selected[index] += 1

        if not args.no_progress:
            p_bar.close()

        scheduler.step()

        if epoch % args.val_freq != 0 and epoch != args.epochs - 1:
            continue

        test_model = model

        # test poisoning success
        test_model.eval()
        # if args.clean:
        #     p_acc = -1
        #     t_acc = -1
        #     pass
        # elif args.backdoor:
        #     p_accs = []
        #     t_accs = []
        #     for t in target_img:
        #         target_conf = torch.softmax(test_model(t.unsqueeze(0).to(args.device)), dim=-1)
        #         target_pred = target_conf.max(1)[1].item()
        #         p_acc = (target_pred == poisoned_label)
        #         t_acc = (target_pred == target_class)
        #
        #         p_accs.append(p_acc)
        #         t_accs.append(t_acc)
        #
        #     p_acc = np.mean(p_accs)
        #     t_acc = np.mean(t_accs)
        # else:
        #     target_conf = torch.softmax(test_model(target_img.unsqueeze(0).to(args.device)), dim=-1)
        #     target_pred = target_conf.max(1)[1].item()
        #     p_acc = (target_pred == poisoned_label)
        #     t_acc = (target_pred == target_class)
        #
        # test_loss, test_acc = test(args, test_loader, test_model, loss_fn, p_acc)

        # print(f"Poison acc: {p_acc}")

        # is_best = test_acc > best_acc
        # best_acc = max(test_acc, best_acc)
        # if is_best:
        #     best_p_acc = p_acc

        # model_to_save = model.module if hasattr(model, "module") else model
        # save_checkpoint({
        #     'epoch': epoch + 1,
        #     'state_dict': model_to_save.state_dict(),
        #     'acc': test_acc,
        #     'best_acc': best_acc,
        #     'poison_acc': p_acc,
        #     'best_poison_acc': best_p_acc,
        #     'optimizer': optimizer.state_dict(),
        #     'scheduler': scheduler.state_dict(),
        #     'times_selected': times_selected,
        # }, is_best, args.out)

        test_bd_mw_accs.append(test(args, x_test[:100000], y_test[:100000], test_model, loss_fn)[1])
        test_clean_gw_accs.append(test(args, x_test[100000:200000], y_test[100000:200000], test_model, loss_fn)[1])
        test_clean_mw_accs.append(test(args, x_test[200000:300000], y_test[200000:300000], test_model, loss_fn)[1])
        # logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
        logger.info('test_bd_mw acc: {:.2f}'.format(test_bd_mw_accs[-1]))
        logger.info(f'test_clean_mw acc: {test_clean_mw_accs[-1]}')
        logger.info(f'test_clean_gw acc: {test_clean_gw_accs[-1]}\n')

    # time_end = time.time()
    # # Save to csv output
    # save_path = os.path.join(args.out, 'result.csv')
    # save_dict = {
    #     'epoch': epoch + 1,
    #     'acc': test_acc,
    #     'best_acc': best_acc,
    #     'poison_acc': p_acc,
    #     'best_poison_acc': best_p_acc,
    #     'mean_poison_acc': np.mean(poison_accs),
    #     'runtime': str(datetime.timedelta(seconds=time_end - time_start)).replace(',', ''),
    # }
    # save_dict = {**save_dict, **(vars(args))}
    #
    # print(f"Saving final results to {save_path}")
    # df = pd.DataFrame.from_dict([save_dict])
    # df.to_csv(save_path, mode='a', index=False)


def test(args, x_test, y_test, model, loss_fn):
    test_loader = TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test).long())
    test_loader = DataLoader(test_loader, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    if not args.no_progress:
        test_loader = tqdm(test_loader)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets).mean()

            prec1 = accuracy(outputs, targets)[0]
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description(
                    "Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. ".format(
                        batch=batch_idx + 1,
                        iter=len(test_loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        top1=top1.avg,
                    ))
        if not args.no_progress:
            test_loader.close()

    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    return losses.avg, top1.avg


def main():
    parser = argparse.ArgumentParser(description='Friendly Noise Defense')
    parser.add_argument('--gpu-id', default='0', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'tinyimagenet'],
                        help='dataset name')
    parser.add_argument('--arch', default='resnet18', type=str,
                        choices=['resnet18', 'alexnet', 'lenet'],
                        help='dataset name')
    parser.add_argument('--epochs', default=40, type=int,
                        help='number of epochs')
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--val_freq', default=10, type=int,
                        help='how frequent to run validation')
    parser.add_argument('--batch-size', default=128, type=int,
                        help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--warmup', default=0, type=int,
                        help='warmup epochs (default: 0)')
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    parser.add_argument('--out', default='result',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=None, type=int,
                        help="random seed")
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")

    # Poison Setting
    parser.add_argument('--clean', action='store_true', help='train with the clean data')
    parser.add_argument("--poisons_path", type=str, help="where are the poisons?")
    parser.add_argument('--dataset_index', type=str, default=None, help='dataset index for naming purposes')
    parser.add_argument("--patch_size", type=int, default=5, help="Size of the patch")
    parser.add_argument('--trigger_path', type=str, default=None, help='path to the trigger')
    parser.add_argument("--backdoor", action='store_true', help='whether we are using backdoor attack')
    parser.add_argument('--scenario', default='scratch', choices=('scratch', 'transfer', 'finetune'),
                        help='select the training setting')

    parser.add_argument('--no_augment', action='store_true', help='no augment')

    parser.add_argument('--noise_type', type=str, nargs='*', help='type of noise to apply', default=[],
                        choices=["uniform", "gaussian", "bernoulli", "gaussian_blur", "friendly"])
    parser.add_argument('--noise_eps', type=float, help='strength of noise to apply', default=8)

    parser.add_argument('--friendly_begin_epoch', type=int, help='epoch to start adding friendly noise', default=0)
    parser.add_argument('--friendly_epochs', type=int, help='number of epochs to run friendly noise generation for',
                        default=30)
    parser.add_argument('--friendly_lr', type=float, help='learning rate for friendly noise generation', default=100)
    parser.add_argument('--friendly_mu', type=float, help='weight of magnitude constraint term in friendly noise loss',
                        default=1)
    parser.add_argument('--friendly_clamp', type=float, help='how much to clamp generated friendly noise', default=16)
    parser.add_argument('--friendly_loss', type=str, help='loss to use for friendly noise', default='KL',
                        choices=['MSE', 'KL'])
    parser.add_argument('--save_friendly_noise', action='store_true', help='save friendly noise')
    parser.add_argument('--load_friendly_noise', type=str, help='load friendly noise', default=None)
    parser.add_argument("--load_poison_dir", default=None, type=str,
                        help="directory containing poisoned data"
                        )
    parser.add_argument("--ember_data_dir", default="/tmp", type=str, help="dir to store cached ember dataset")
    args = parser.parse_args()

    global best_acc, transform_train, transform_val
    transform_train = []

    if "uniform" in args.noise_type:
        print(f"Adding uniform noise: {args.noise_eps}")
        transform_train.append(UniformNoise(eps=args.noise_eps / MAX_VALUE))
    if "gaussian" in args.noise_type:
        print(f"Adding gaussian noise: {args.noise_eps}")
        transform_train.append(GaussianNoise(eps=args.noise_eps / MAX_VALUE))
    if "bernoulli" in args.noise_type:
        print(f"Adding bernoulli noise: {args.noise_eps}")
        transform_train.append(BernoulliNoise(eps=args.noise_eps / MAX_VALUE))
    if "friendly" in args.noise_type:
        print(f"Using friendly noise")
    transform_train = transforms.Compose(transform_train)

    device = torch.device('cuda', args.gpu_id)
    args.world_size = 1
    args.n_gpu = torch.cuda.device_count()

    args.device = device

    if args.seed is not None:
        set_seed(args)

    if args.epochs == 200:
        args.steps = [int(args.epochs * 1 / 2), int(args.epochs * 3 / 4)]
    else:
        args.steps = [args.epochs // 2.667, args.epochs // 1.6, args.epochs // 1.142]

    dir_name = f'{args.arch}-{args.dataset}'
    dir_name += '-clean' if args.clean else f'-{"ember"}'
    dir_name += f'.{args.seed}-sl-epoch{args.epochs}'
    dir_name += f'-warmup{args.warmup}' if args.warmup > 0 else ''
    dir_name += f'-{args.scenario}' if args.scenario != 'scratch' else ''
    dir_name += f'-noaug' if args.no_augment else ''
    dir_name += f'-{"-".join(args.noise_type)}' if len(args.noise_type) != 0 else ''
    dir_name += f'-p{args.friendly_begin_epoch}' if 'friendly' in args.noise_type else ''
    dir_name += f'-_{args.friendly_epochs}' if 'friendly' in args.noise_type else ''
    dir_name += f'-_{args.friendly_lr}' if 'friendly' in args.noise_type else ''
    dir_name += f'-_{args.friendly_mu}' if 'friendly' in args.noise_type else ''
    dir_name += f'-_clp{args.friendly_clamp}' if 'friendly' in args.noise_type else ''
    dir_name += f'-_loss-{args.friendly_loss}' if 'friendly' in args.noise_type else ''
    args.out = os.path.join(args.out, dir_name)

    os.makedirs(args.out, exist_ok=True)

    # write and save training log
    logging.basicConfig(
        filename=f"{args.out}/output.log",
        filemode='a',
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    logger.warning(
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}")

    logger.info(dict(args._get_kwargs()))
    data_processor = EmberPoisonDataPreProcessor(args)
    minmax = MinMaxScaler(clip=True)
    data_processor.x_train = minmax.fit_transform(data_processor.x_train)
    # data_processor.x_train = np.expand_dims(data_processor.x_train, axis=(1, 2))
    data_processor.x_test = minmax.transform(data_processor.x_test)
    # data_processor.x_test = np.expand_dims(data_processor.x_test, axis=(1, 2))
    base_dataset = TensorDataset(torch.Tensor(data_processor.x_train), torch.Tensor(data_processor.y_train).long())
    args.train_size = data_processor.x_train.shape[0]

    train_dataset = PerturbedPoisonedDataset(
        trainset=base_dataset,
        indices=np.array(range(len(base_dataset))),
        transform=transform_train,
        return_index=True,
        size=args.train_size,
        convert_tensor=False)

    noaug_train_dataset = PerturbedPoisonedDataset(
        trainset=base_dataset,
        indices=np.array(range(len(base_dataset))),
        transform=None,
        return_index=True,
        size=args.train_size,
        convert_tensor=False)

    poison_indices = []
    poison_tuples = []
    target_class = 1
    poisoned_label = 0
    target_img = None

    model = mlp_4layer(2351, 1)
    model.to(args.device)

    logger.info(f"Target class {target_class}; Poisoned label: {poisoned_label}")

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False)

    noaug_train_loader = DataLoader(
        noaug_train_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False)

    test_loader = (data_processor.x_test, data_processor.y_test)

    if args.scenario != 'scratch':
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])
        if args.scenario == 'transfer':
            logger.info("==> Freezing the feature representation..")
            for param in model.parameters():
                param.requires_grad = False
        else:
            logger.info("==> Decreasing the learning rate for fine-tuning..")
            args.lr = 1e-4
        logger.info("==> Reinitializing the classifier..")
        num_ftrs = model.linear.in_features
        model.linear = nn.Linear(num_ftrs, args.num_classes).to(args.device)  # requires grad by default

    no_decay = ['bias', 'bn']
    grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.steps)
    if args.warmup > 0:
        logger.info('Warm start learning rate')
        raise NotImplementedError
        # lr_scheduler_f = GradualWarmupScheduler(optimizer, 1.0, args.warmup, scheduler)
    else:
        logger.info('No Warm start')
        lr_scheduler_f = scheduler

    loss_fn = nn.CrossEntropyLoss(reduction='none')

    args.start_epoch = 0

    if args.resume and (args.scenario == 'scratch'):
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume, map_location='cpu')
        best_acc = checkpoint['best_acc']
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

    if args.load_friendly_noise:
        logger.info(f"Loading friendly noise from {args.load_friendly_noise}...")
        perturbs = np.load(args.load_friendly_noise)
        train_loader.dataset.set_perturbations(torch.Tensor(perturbs))
        logger.info(f"Friendly noise loaded!")

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(
        f"  Total train batch size = {args.batch_size * args.world_size}")

    model.zero_grad()
    model_to_save = model.module if hasattr(model, "module") else model
    # save_checkpoint({
    #     'epoch': 0,
    #     'state_dict': model_to_save.state_dict(),
    #     'optimizer': optimizer.state_dict(),
    #     'scheduler': scheduler.state_dict(),
    # }, is_best=False, checkpoint=args.out, filename='init.pth.tar')
    train(args, train_loader, noaug_train_loader, test_loader, model, optimizer, lr_scheduler_f, target_img,
          target_class, poisoned_label, None, loss_fn, None, None, None)


if __name__ == '__main__':
    main()
