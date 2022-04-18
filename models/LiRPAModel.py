from abc import ABC, abstractmethod

from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import *
from auto_LiRPA.eps_scheduler import LinearScheduler, AdaptiveScheduler, SmoothedScheduler, FixedScheduler
from auto_LiRPA.utils import MultiAverageMeter


class LiRPAModel(ABC):
    def __init__(self, n_features, n_classes, args, device, model_ori, lr=1e-3):
        self.input_shape = n_features
        self.n_classes = n_classes
        self.lr = lr
        self.args = args
        self.device = device
        self.init_model = model_ori
        self.init()

    def init(self):
        ## Step 1: Initial original model as usual
        self.model_ori = self.init_model(in_ch=self.input_shape[-1], in_dim=self.input_shape[0])

        ## Step 3: wrap model with auto_LiRPA
        # The second parameter dummy_input is for constructing the trace of the computational graph.
        dummy_input = torch.randn(*([2] + list(self.input_shape)))
        self.model = BoundedModule(self.model_ori, dummy_input,
                                   bound_opts={'relu': self.args.bound_opts, 'conv_mode': self.args.conv_mode},
                                   device=self.device)

        # print("Model structure: \n", str(self.model_ori))

    def fit(self, X, y, batch_size, epochs, data_aug=None):
        data = TensorDataset(torch.Tensor(X), torch.Tensor(y).long().max(dim=-1)[1])
        loader = DataLoader(data, batch_size=batch_size, shuffle=True, pin_memory=True)

        ## Step 4 prepare optimizer, epsilon scheduler and learning rate scheduler
        eps_scheduler = eval(self.args.scheduler_name)(self.args.eps, self.args.scheduler_opts)
        opt = optim.Adam(self.model.parameters(), lr=self.lr)
        lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)

        for t in range(1, epochs + 1):
            if eps_scheduler.reached_max_eps():
                # Only decay learning rate after reaching the maximum eps
                lr_scheduler.step()
            self.train(eps_scheduler, opt, loader, data_aug)

    def evaluate(self, x_test, y_test):
        data = TensorDataset(torch.Tensor(x_test),
                             torch.Tensor(y_test).long().max(dim=-1)[1])
        loader = DataLoader(data, batch_size=256, shuffle=False, pin_memory=True)

        eps_scheduler = FixedScheduler(self.args.eps)
        with torch.no_grad():
            predictions, verified = self.test(eps_scheduler, loader)
            print("Test accuracy: ", np.mean(predictions == np.argmax(y_test, axis=-1)))
            predictions = predictions * verified + (1 - verified) * self.n_classes
            print("Certified accuracy: ", np.mean(predictions == np.argmax(y_test, axis=-1)))
            return predictions

    def test(self, eps_scheduler, loader):
        norm = float(self.args.norm)

        self.model.eval()
        eps_scheduler.eval()
        predictions = np.array([], dtype=np.int)
        verified = np.array([], dtype=np.bool)

        for i, (data, _) in enumerate(loader):
            eps_scheduler.step_batch()
            eps = eps_scheduler.get_eps()

            # bound input for Linf norm used only
            data_ub = data_lb = data

            if list(self.model.parameters())[0].is_cuda:
                data, labels = data.cuda(), labels.cuda()
                data_lb, data_ub = data_lb.cuda(), data_ub.cuda()

            # Specify Lp norm perturbation.
            # When using Linf perturbation, we manually set element-wise bound x_L and x_U. eps is not used for Linf norm.
            if norm > 0:
                ptb = PerturbationLpNorm(norm=norm, eps=eps, x_L=data_lb, x_U=data_ub)
            elif norm == 0:
                ptb = PerturbationL0Norm(eps=eps_scheduler.get_max_eps(),
                                         ratio=eps_scheduler.get_eps() / eps_scheduler.get_max_eps())
            x = BoundedTensor(data, ptb)

            output = self.model(x)
            batch_predictions = torch.argmax(output, dim=1)
            # meter.update('CE', regular_ce.item(), x.size(0))
            # meter.update('Err', torch.sum(torch.argmax(output, dim=1) != labels).cpu().detach().numpy() / x.size(0),
            #              x.size(0))
            # generate specifications
            c = torch.eye(self.n_classes).type_as(data)[batch_predictions].unsqueeze(1) - torch.eye(
                self.n_classes).type_as(
                data).unsqueeze(0)
            # remove specifications to self
            I = (~(batch_predictions.data.unsqueeze(1) == torch.arange(self.n_classes).type_as(
                batch_predictions.data).unsqueeze(0)))
            c = (c[I].view(data.size(0), self.n_classes - 1, self.n_classes))
            if list(self.model.parameters())[0].is_cuda:
                c = c.cuda()

            if self.args.bound_type == "IBP":
                lb, ub = self.model.compute_bounds(IBP=True, C=c, method=None)
            elif self.args.bound_type == "CROWN":
                lb, ub = self.model.compute_bounds(IBP=False, C=c, method="backward", bound_upper=False)
            elif self.args.bound_type == "CROWN-IBP":
                # lb, ub = model.compute_bounds(ptb=ptb, IBP=True, x=data, C=c, method="backward")  # pure IBP bound
                # we use a mixed IBP and CROWN-IBP bounds, leading to better performance (Zhang et al., ICLR 2020)
                factor = (eps_scheduler.get_max_eps() - eps) / eps_scheduler.get_max_eps()
                ilb, iub = self.model.compute_bounds(IBP=True, C=c, method=None)
                if factor < 1e-5:
                    lb = ilb
                else:
                    clb, cub = self.model.compute_bounds(IBP=False, C=c, method="backward", bound_upper=False)
                    lb = clb * factor + ilb * (1 - factor)
            elif self.args.bound_type == "CROWN-FAST":
                # Similar to CROWN-IBP but no mix between IBP and CROWN bounds.
                lb, ub = self.model.compute_bounds(IBP=True, C=c, method=None)
                lb, ub = self.model.compute_bounds(IBP=False, C=c, method="backward", bound_upper=False)

            # Pad zero at the beginning for each example, and use fake label "0" for all examples
            # lb_padded = torch.cat((torch.zeros(size=(lb.size(0), 1), dtype=lb.dtype, device=lb.device), lb), dim=1)
            # fake_labels = torch.zeros(size=(lb.size(0),), dtype=torch.int64, device=lb.device)
            # robust_ce = CrossEntropyLoss()(-lb_padded, fake_labels)
            # meter.update('Loss', loss.item(), data.size(0))
            # if batch_method != "natural":
            #     meter.update('Robust_CE', robust_ce.item(), data.size(0))
            #     # For an example, if lower bounds of margins is >0 for all classes, the output is verifiably correct.
            #     # If any margin is < 0 this example is counted as an error
            #     meter.update('Verified_Err', torch.sum((lb < 0).any(dim=1)).item() / data.size(0), data.size(0))
            # if i % 50 == 0 and train:
            #     print('[{:4d}]: eps={:.8f} {}'.format(i, eps, meter))
            batch_predictions = batch_predictions.numpy()
            batch_verified = (lb > 0).all(dim=1).numpy()
            predictions = np.append(predictions, batch_predictions)
            verified = np.append(verified, batch_verified)

        # print('[{:4d}]: eps={:.8f} {}'.format(i, eps, meter))
        return predictions, verified

    def train(self, eps_scheduler, opt, loader, data_aug):
        norm = float(self.args.norm)

        meter = MultiAverageMeter()
        self.model.train()
        eps_scheduler.train()
        eps_scheduler.step_epoch()
        eps_scheduler.set_epoch_length(int((len(loader.dataset) + loader.batch_size - 1) / loader.batch_size))

        for i, (data, labels) in enumerate(loader):
            eps_scheduler.step_batch()
            eps = eps_scheduler.get_eps()
            # For small eps just use natural training, no need to compute LiRPA bounds
            batch_method = "robust"
            if eps < 1e-20:
                batch_method = "natural"
            opt.zero_grad()
            # generate specifications
            c = torch.eye(self.n_classes).type_as(data)[labels].unsqueeze(1) - torch.eye(self.n_classes).type_as(
                data).unsqueeze(0)
            # remove specifications to self
            I = (~(labels.data.unsqueeze(1) == torch.arange(self.n_classes).type_as(labels.data).unsqueeze(0)))
            c = (c[I].view(data.size(0), self.n_classes - 1, self.n_classes))
            # bound input for Linf norm used only
            if data_aug is not None:
                data = data_aug(data)
            data_ub = data_lb = data

            if list(self.model.parameters())[0].is_cuda:
                data, labels, c = data.cuda(), labels.cuda(), c.cuda()
                data_lb, data_ub = data_lb.cuda(), data_ub.cuda()

            # Specify Lp norm perturbation.
            # When using Linf perturbation, we manually set element-wise bound x_L and x_U. eps is not used for Linf norm.
            if norm > 0:
                ptb = PerturbationLpNorm(norm=norm, eps=eps, x_L=data_lb, x_U=data_ub)
            elif norm == 0:
                ptb = PerturbationL0Norm(eps=eps_scheduler.get_max_eps(),
                                         ratio=eps_scheduler.get_eps() / eps_scheduler.get_max_eps())
            x = BoundedTensor(data, ptb)

            output = self.model(x)
            regular_ce = CrossEntropyLoss()(output, labels)  # regular CrossEntropyLoss used for warming up
            meter.update('CE', regular_ce.item(), x.size(0))
            meter.update('Err', torch.sum(torch.argmax(output, dim=1) != labels).cpu().detach().numpy() / x.size(0),
                         x.size(0))

            if batch_method == "robust":
                if self.args.bound_type == "IBP":
                    lb, ub = self.model.compute_bounds(IBP=True, C=c, method=None)
                elif self.args.bound_type == "CROWN":
                    lb, ub = self.model.compute_bounds(IBP=False, C=c, method="backward", bound_upper=False)
                elif self.args.bound_type == "CROWN-IBP":
                    # lb, ub = model.compute_bounds(ptb=ptb, IBP=True, x=data, C=c, method="backward")  # pure IBP bound
                    # we use a mixed IBP and CROWN-IBP bounds, leading to better performance (Zhang et al., ICLR 2020)
                    factor = (eps_scheduler.get_max_eps() - eps) / eps_scheduler.get_max_eps()
                    ilb, iub = self.model.compute_bounds(IBP=True, C=c, method=None)
                    if factor < 1e-5:
                        lb = ilb
                    else:
                        clb, cub = self.model.compute_bounds(IBP=False, C=c, method="backward", bound_upper=False)
                        lb = clb * factor + ilb * (1 - factor)
                elif self.args.bound_type == "CROWN-FAST":
                    # Similar to CROWN-IBP but no mix between IBP and CROWN bounds.
                    lb, ub = self.model.compute_bounds(IBP=True, C=c, method=None)
                    lb, ub = self.model.compute_bounds(IBP=False, C=c, method="backward", bound_upper=False)

                # Pad zero at the beginning for each example, and use fake label "0" for all examples
                lb_padded = torch.cat((torch.zeros(size=(lb.size(0), 1), dtype=lb.dtype, device=lb.device), lb), dim=1)
                fake_labels = torch.zeros(size=(lb.size(0),), dtype=torch.int64, device=lb.device)
                robust_ce = CrossEntropyLoss()(-lb_padded, fake_labels)
            if batch_method == "robust":
                loss = robust_ce
            elif batch_method == "natural":
                loss = regular_ce
            loss.backward()
            eps_scheduler.update_loss(loss.item() - regular_ce.item())
            opt.step()
            meter.update('Loss', loss.item(), data.size(0))
            if batch_method != "natural":
                meter.update('Robust_CE', robust_ce.item(), data.size(0))
                #     # For an example, if lower bounds of margins is >0 for all classes, the output is verifiably correct.
                #     # If any margin is < 0 this example is counted as an error
                meter.update('Verified_Err', torch.sum((lb < 0).any(dim=1)).item() / data.size(0), data.size(0))
            # # if i % 50 == 0 and train:
            #     print('[{:4d}]: eps={:.8f} {}'.format(i, eps, meter))

        # print('[{:4d}]: eps={:.8f} {}'.format(i, eps, meter))
