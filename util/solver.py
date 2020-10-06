"""
Performs training, validation, testing generically for any classification model and calculates loss/accuracy and saves it to tensorboard.

Author: Lukas Hoellein
"""

import numpy as np

import torch
from models.losses.TripletLoss import TripletLoss
from models.losses.metrics import Top_K_Accuracy
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

def wrap_data(xb, yb, device):
    xb, yb = Variable(xb), Variable(yb)
    if str(device) != 'cpu':
        xb, yb = xb.cuda(), yb.cuda()

    return xb, yb

class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    default_scheduler_args = {"step_size": 1,
                              "gamma": 0.3}

    def __init__(self,
                 triplets_as_batches,
                 outputs_as_triplets,
                 loss_func=TripletLoss(),
                 scheduler=torch.optim.lr_scheduler.StepLR,
                 scheduler_args={},
                 top_k_accs = [1, 5],
                 optim=torch.optim.Adam,
                 optim_args={},
                 extra_args={},
                 log_dir=None):
        """

        Parameters
        ----------
        optim: which optimizer to use, e.g. Adam
        optim_args: see also default_adam_args: specify here valid dictionary of arguments for chosen optimizer
        extra_args: extra_args that should be used when logging to tensorboard (e.g. model hyperparameters)
        loss_func: loss function, e.g. Cross-Entropy-Loss
        sample_loader: function on how to return input ('x') and target ('y') from a mini-batch sample.
                       This way the solver can work for any data by just defining how to get input and target through the caller.
                       The output of the function must satisfy the following format: { 'x': sample_input, 'y': sample_targets }
                       (default: returns sample['x'] and sample['y'])
        acc_func: how to calculate accuracy measure between scores and y. default: Accuracy for class prediction via CrossEntropyLoss
        log_dir: where to log to tensorboard
        """

        # init optim with default args and given args
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim

        # init scheduler with default args and given args
        if scheduler == torch.optim.lr_scheduler.StepLR:
            scheduler_args_merged = self.default_scheduler_args.copy()
            scheduler_args_merged.update(scheduler_args)
        else:
            scheduler_args_merged = scheduler_args
        self.scheduler = scheduler
        self.scheduler_args = scheduler_args_merged

        # init loss and acc
        self.loss_func = loss_func
        self.acc_func = Top_K_Accuracy(top_k_accs)

        # init conversion functions for faster batch-processing
        self.triplets_as_batches = triplets_as_batches
        self.outputs_as_triplets = outputs_as_triplets

        # init tensorboard log + hparam infos
        self.writer = SummaryWriter(log_dir)

        for key in extra_args.keys():
            extra_args[key] = str(extra_args[key])

        scheduler_args_str = {k: str(v) for k, v in self.scheduler_args.items()}

        self.hparam_dict = {'loss function': type(self.loss_func).__name__,
                            'triplet_loss_margin': self.loss_func.margin,
                            'triplet_loss_reduction_mode': self.loss_func.reduction_mode,
                            'optimizer': self.optim.__name__,
                            'learning rate': self.optim_args['lr'],
                            'scheduler': self.scheduler.__name__,
                            'weight_decay': self.optim_args['weight_decay'],
                            **scheduler_args_str,
                            **extra_args}

        print("Hyperparameters of this solver: {}".format(self.hparam_dict))

        self._reset_histories()

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_pos_dist_history = []
        self.train_neg_dist_history = []
        self.val_loss_history = []
        self.val_pos_dist_history = []
        self.val_neg_dist_history = []

    def forward_pass(self, model, sample, device):

        sample = self.triplets_as_batches(sample, 1)
        if torch.cuda.is_available():
            sample["image"] = sample["image"].to(device)

        encodings = model(sample)
        encodings = self.outputs_as_triplets(encodings, 1)

        loss, pos_dist, neg_dist = self.loss_func(encodings)

        return loss, pos_dist, neg_dist

    def train_one_epoch(self, model, tqdm_mode, train_loader, verbose, optim, epoch, iter_per_epoch, device, log_nth_iter, log_nth_epoch, num_epochs):
        model.train()  # TRAINING mode (for dropout, batchnorm, etc.)
        train_losses = []
        train_pos_dists = []
        train_neg_dists = []

        train_minibatches = train_loader
        if tqdm_mode == 'epoch':
            train_minibatches = tqdm(train_minibatches)

        # MEASURE ELAPSED TIME
        if verbose:
            # start first dataloading record
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

        for i, sample in enumerate(train_minibatches):  # for every minibatch in training set
            # MEASURE ELAPSED TIME
            if verbose:
                # end dataloading pass record
                end.record()
                torch.cuda.synchronize()
                print("Dataloading took: {}".format(start.elapsed_time(end)))

                # start forward/backward record
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

            # FORWARD PASS --> Loss + acc calculation
            # print("Time until next forward pass (loading from dataloader + backward pass) took: {}".format(time() - start))
            train_loss, train_pos_dist, train_neg_dist = self.forward_pass(model, sample, device)
            # start = time()

            # BACKWARD PASS --> Gradient-Descent update
            train_loss.backward()
            optim.step()
            optim.zero_grad()

            # LOGGING of loss and accuracy
            train_loss = train_loss.data.cpu().numpy()
            train_losses.append(train_loss)
            train_pos_dists.append(train_pos_dist)
            train_neg_dists.append(train_neg_dist)

            self.writer.add_scalar('Batch/Loss/Train', train_loss, i + epoch * iter_per_epoch)
            self.writer.add_scalar(f'Batch/Pos_Dist/Train', train_pos_dist, i + epoch * iter_per_epoch)
            self.writer.add_scalar(f'Batch/Neg_Dist/Train', train_neg_dist, i + epoch * iter_per_epoch)
            self.writer.flush()

            # Print loss every log_nth iteration
            if log_nth_iter != 0 and (i + 1) % log_nth_iter == 0:
                print("[Iteration {cur}/{max}] TRAIN loss: {loss}".format(cur=i + 1,
                                                                          max=iter_per_epoch,
                                                                          loss=train_loss))

            # MEASURE ELAPSED TIME
            if verbose:
                # end forward/backward pass record
                end.record()
                torch.cuda.synchronize()
                print("Forward/Backward Pass took: {}".format(start.elapsed_time(end)))

                # start dataloading record
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()

        # ONE EPOCH PASSED --> calculate + log mean train accuracy/loss for this epoch
        mean_train_loss = np.mean(train_losses)
        self.train_loss_history.append(mean_train_loss)
        self.writer.add_scalar('Epoch/Loss/Train', mean_train_loss, epoch)

        mean_train_pos_dist = np.mean(train_pos_dists)
        self.train_pos_dist_history.append(mean_train_pos_dist)
        self.writer.add_scalar('Epoch/Pos_Dist/Train', mean_train_pos_dist, epoch)

        mean_train_neg_dist = np.mean(train_neg_dists)
        self.train_neg_dist_history.append(mean_train_neg_dist)
        self.writer.add_scalar('Epoch/Neg_Dist/Train', mean_train_neg_dist, epoch)

        if log_nth_epoch != 0 and (epoch + 1) % log_nth_epoch == 0:
            print("[EPOCH {cur}/{max}] TRAIN mean loss / pos_dist / neg_dist: {loss}, {pos_dist}, {neg_dist}".format(
                cur=epoch + 1,
                max=num_epochs,
                loss=mean_train_loss,
                pos_dist=mean_train_pos_dist,
                neg_dist=mean_train_neg_dist))

        return mean_train_loss, mean_train_pos_dist, mean_train_neg_dist

    def val_one_epoch(self, model, tqdm_mode, val_loader, epoch, device, log_nth_iter, log_nth_epoch, num_epochs):
        # ONE EPOCH PASSED --> calculate + log validation accuracy/loss for this epoch
        model.eval()  # EVAL mode (for dropout, batchnorm, etc.)
        with torch.no_grad():
            val_losses = []
            val_pos_dists = []
            val_neg_dists = []

            val_minibatches = val_loader
            if tqdm_mode == 'epoch':
                val_minibatches = tqdm(val_minibatches)

            for i, sample in enumerate(val_minibatches):
                # FORWARD PASS --> Loss + acc calculation
                val_loss, val_pos_dist, val_neg_dist = self.forward_pass(model, sample, device)

                # LOGGING of loss and accuracy
                val_loss = val_loss.data.cpu().numpy()
                val_losses.append(val_loss)
                val_pos_dists.append(val_pos_dist)
                val_neg_dists.append(val_neg_dist)

                self.writer.add_scalar('Batch/Loss/Val', val_loss, i + epoch * len(val_loader))
                self.writer.add_scalar(f'Batch/Pos_Dist/Val', val_pos_dist, i + epoch * len(val_loader))
                self.writer.add_scalar(f'Batch/Neg_Dist/Val', val_neg_dist, i + epoch * len(val_loader))
                self.writer.flush()

                # Print loss every log_nth iteration
                if log_nth_iter != 0 and (i + 1) % log_nth_iter == 0:
                    print("[Iteration {cur}/{max}] Val loss: {loss}".format(cur=i + 1,
                                                                            max=len(val_loader),
                                                                            loss=val_loss))

            mean_val_loss = np.mean(val_losses)
            self.val_loss_history.append(mean_val_loss)
            self.writer.add_scalar('Epoch/Loss/Val', mean_val_loss, epoch)

            mean_val_pos_dist = np.mean(val_pos_dists)
            self.val_pos_dist_history.append(mean_val_pos_dist)
            self.writer.add_scalar('Epoch/Pos_Dist/Val', mean_val_pos_dist, epoch)

            mean_val_neg_dist = np.mean(val_neg_dists)
            self.val_neg_dist_history.append(mean_val_neg_dist)
            self.writer.add_scalar('Epoch/Neg_Dist/Val', mean_val_neg_dist, epoch)

            self.writer.flush()

            if log_nth_epoch != 0 and (epoch + 1) % log_nth_epoch == 0:
                print(
                    "[EPOCH {cur}/{max}] VAL mean mean loss / pos_dist / neg_dist: {loss}, {pos_dist}, {neg_dist}".format(
                        cur=epoch + 1,
                        max=num_epochs,
                        loss=mean_val_loss,
                        pos_dist=mean_val_pos_dist,
                        neg_dist=mean_val_neg_dist))

            return mean_val_loss, mean_val_pos_dist, mean_val_neg_dist

    def train(self, model, train_loader, val_loader, start_epoch=0, num_epochs=10, log_nth_iter=1, log_nth_epoch=1, tqdm_mode='total', verbose=False):
        """
        Train a given model with the provided data.

        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """

        # model to cuda before optim creation: https://pytorch.org/docs/stable/optim.html
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # optim creation, scheduler creation
        optim = self.optim(filter(lambda p: p.requires_grad, model.parameters()), **self.optim_args)
        scheduler = self.scheduler(optim, **self.scheduler_args)

        # init loss / acc history and train length
        self._reset_histories()
        iter_per_epoch = len(train_loader)

        # start training
        print('START TRAIN on device: {}'.format(device))

        max_epoch = start_epoch + num_epochs
        epochs = range(start_epoch, max_epoch)
        if tqdm_mode == 'total':
            epochs = tqdm(range(start_epoch, max_epoch))

        # epoch loop
        for epoch in epochs:

            # train iterations for one epoch
            mean_train_loss, mean_train_pos_dist, mean_train_neg_dist = self.train_one_epoch(model,
                                                                                             tqdm_mode,
                                                                                             train_loader,
                                                                                             verbose,
                                                                                             optim,
                                                                                             epoch,
                                                                                             iter_per_epoch,
                                                                                             device,
                                                                                             log_nth_iter,
                                                                                             log_nth_epoch,
                                                                                             num_epochs)

            # val iterations for one epoch
            mean_val_loss, mean_val_pos_dist, mean_val_neg_dist = self.val_one_epoch(model,
                                                                                     tqdm_mode,
                                                                                     val_loader,
                                                                                     epoch,
                                                                                     device,
                                                                                     log_nth_iter,
                                                                                     log_nth_epoch,
                                                                                     num_epochs)

            # log after one epoch
            self.writer.add_scalars('Epoch/Loss',
                                    {'train': mean_train_loss,
                                     'val': mean_val_loss},
                                    epoch)

            self.writer.add_scalars('Epoch/Pos_Dist',
                                    {'train': mean_train_pos_dist,
                                     'val': mean_val_pos_dist},
                                    epoch)

            self.writer.add_scalars('Epoch/Neg_Dist',
                                    {'train': mean_train_neg_dist,
                                     'val': mean_val_neg_dist},
                                    epoch)

            self.writer.add_scalars('Epoch/Dist',
                                    {'pos_val': mean_val_pos_dist,
                                     'neg_val': mean_val_neg_dist},
                                    epoch)

            # Decay Learning Rate
            # Currently, only the ReduceLROnPlateau scheduler needs an argument (last val loss).
            # ALl others are a non-argument call to step() method.
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(self.val_loss_history[-1])
            else:
                scheduler.step()

        self.finish()

    def finish(self):
        self.writer.add_hparams(self.hparam_dict, {
            'HParam/Pos_Dist/Val': self.val_pos_dist_history[-1],
            'HParam/Pos_Dist/Train': self.train_pos_dist_history[-1],
            'HParam/Neg_Dist/Val': self.val_neg_dist_history[-1],
            'HParam/Neg_Dist/Train': self.train_neg_dist_history[-1],
            'HParam/Loss/Val': self.val_loss_history[-1],
            'HParam/Loss/Train': self.train_loss_history[-1]
        })
        self.writer.flush()
        print('FINISH.')
