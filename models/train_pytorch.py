import torch
import torch.optim as optim
import numpy as np
from copy import deepcopy
from models.utils import validate_pytorch


def restore_parameters(model, best_model):
    '''Move parameter values from best_model to model.'''
    for param, best_param in zip(model.parameters(), best_model.parameters()):
        param.data = best_param


class AverageMeter(object):
    '''
    For tracking moving average of loss.

    Args:
      r: parameter for calcualting exponentially moving average.
    '''
    def __init__(self, r=0.1):
        self.r = r
        self.reset()

    def reset(self):
        self.loss = None

    def update(self, loss):
        if not self.loss:
            self.loss = loss
        else:
            self.loss = self.r * self.loss + (1 - self.r) * loss

    def get(self):
        return self.loss


class TrainPyTorch:
    def __init__(self, model):
        self.model = model
        self.train_list = []
        self.val_list = []

    def train(self,
              train_loader,
              val_loader,
              lr,
              mbsize,
              nepochs,
              loss_fn,
              check_every,
              lookback=5,
              evaluation_loss_fn=None,
              verbose=True):

        # For optimization.
        model = self.model
        optimizer = optim.Adam(model.parameters(), lr=lr)
        min_criterion = np.inf
        min_epoch = 0
        meter = AverageMeter()

        # Set up mbsize and check_every.
        train_loader.batch_sampler.batch_size = mbsize
        train_loader.batch_sampler.sampler._num_samples = mbsize * check_every

        if evaluation_loss_fn is None:
            evaluation_loss_fn = loss_fn

        # Determine device.
        device = next(model.parameters()).device

        for epoch in range(nepochs):
            for x, y in train_loader:
                # Move to GPU.
                x = x.to(device=device)
                y = y.to(device=device)

                # Take gradient step.
                loss = loss_fn(model(x), y)
                loss.backward()
                optimizer.step()
                model.zero_grad()
                meter.update(loss.item())

            # Check progress.
            with torch.no_grad():
                train_loss = meter.get()
                meter.reset()
                val_loss = validate_pytorch(
                    model, val_loader, evaluation_loss_fn).item()
                self.val_list.append(val_loss)

                if verbose:
                    print('{}Epoch = {}{}'.format(
                        '-' * 10, epoch + 1, '-' * 10))
                    print('Train performance = {:.4f}'.format(train_loss))
                    print('Val performance = {:.4f}'.format(val_loss))

                # Check convergence criterion.
                if val_loss < min_criterion:
                    min_criterion = val_loss
                    min_epoch = epoch
                    best_model = deepcopy(model)
                elif (epoch - min_epoch) == lookback:
                    if verbose:
                        print('Stopping early')
                    break

        # Restore parameters of best model.
        restore_parameters(model, best_model)

