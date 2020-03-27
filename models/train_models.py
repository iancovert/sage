import torch
import torch.optim as optim
import numpy as np
import sklearn
from copy import deepcopy
from models.utils import restore_parameters, AverageMeter
from models.utils import SklearnClassifierWrapper


def validate(model, loader, loss_fn):
    '''Calculate average loss.'''
    device = next(model.parameters()).device
    mean_loss = 0
    N = 0
    for x, y in loader:
        x = x.to(device=device)
        y = y.to(device=device)
        n = len(x)
        loss = loss_fn(model(x), y)
        mean_loss = (N * mean_loss + n * loss) / (N + n)
        N += n
    return mean_loss


def validate_sklearn(model, loader, loss_fn):
    '''Calculate average loss.'''
    mean_loss = 0
    N = 0

    # Add wrapper if necessary.
    if isinstance(model, sklearn.base.ClassifierMixin):
        model = SklearnClassifierWrapper(model)

    for x, y in loader:
        n = len(x)
        loss = loss_fn(model.predict(x.cpu().numpy()), y.cpu().numpy())
        mean_loss = (N * mean_loss + n * loss) / (N + n)
        N += n
    return mean_loss


class Train:
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
                val_loss = validate(
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

