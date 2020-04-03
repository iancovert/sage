import torch
import torch.nn as nn
import numpy as np
import sklearn


def validate_pytorch(model, loader, loss_fn):
    '''Calculate average loss.

    Args:
      model: PyTorch model. Must be callable, likely inherits from nn.Module.
      loader: PyTorch data loader.
      loss_fn: loss function, such as nn.CrossEntropyLoss().
    '''
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
    '''Calculate average loss.

    Args:
      model: sklearn model.
      loader: PyTorch data loader.
      loss_fn: loss function, such as CrossEntropyLossNP() (a custom function
       in this library) or sklearn.metrics.log_loss.
    '''
    mean_loss = 0
    N = 0

    # Add wrapper if necessary.
    if isinstance(model, sklearn.base.ClassifierMixin):
        model = SklearnClassifierWrapper(model)

    # Detect if one of our custom loss functions.
    custom = (
        isinstance(loss_fn, CrossEntropyLossNP) or
        isinstance(loss_fn, AccuracyNP) or
        isinstance(loss_fn, MSELossNP))

    for x, y in loader:
        n = len(x)
        if custom:
            loss = loss_fn(model.predict(x.cpu().numpy()), y.cpu().numpy())
        else:
            loss = loss_fn(y.cpu().numpy(), model.predict(x.cpu().numpy()))
        mean_loss = (N * mean_loss + n * loss) / (N + n)
        N += n
    return mean_loss


class MSELoss(nn.Module):
    '''MSE loss that always sums over non-batch dimensions. For use with PyTorch
    models only.'''
    def __init__(self, reduction='mean'):
        super(MSELoss, self).__init__()
        assert reduction in ('none', 'mean')
        self.reduction = reduction

    def forward(self, pred, target):
        loss = torch.sum(
            torch.reshape((pred - target) ** 2, (len(pred), -1)),
            dim=1)
        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return loss


class CrossEntropyLoss(nn.Module):
    '''Cross entropy loss that expects probabilities. For use with PyTorch
    models only.'''
    def __init__(self, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()
        assert reduction in ('none', 'mean')
        self.reduction = reduction

    def forward(self, pred, target):
        loss = - torch.log(pred[torch.arange(len(pred)), target])
        if self.reduction == 'mean':
            return torch.mean(loss)
        else:
            return loss


class Accuracy(nn.Module):
    '''0-1 loss. For use with PyTorch models only.'''
    def __init__(self):
        super(Accuracy, self).__init__()

    def forward(self, pred, target):
        '''Calculate 0-1 accuracy.'''
        return (torch.argmax(pred, dim=1) == target).float().mean()


class NegAccuracy(Accuracy):
    '''0-1 loss, negated for use as validation loss during training. For use
    with PyTorch models only.'''
    def __init__(self):
        super(NegAccuracy, self).__init__()

    def forward(self, pred, target):
        '''Negative accuracy, for usage as loss function.'''
        return - super(NegAccuracy, self).forward(pred, target)


class MSELossNP:
    '''MSE loss that always sums over non-batch dimensions. For use with sklearn
    models.'''
    def __init__(self, reduction='mean'):
        assert reduction in ('none', 'mean')
        self.reduction = reduction

    def __call__(self, pred, target):
        # Add dimension to tail of pred, if necessary.
        if target.shape[-1] == 1 and len(target.shape) - len(pred.shape) == 1:
            pred = np.expand_dims(pred, -1)
        loss = np.sum(
            np.reshape((pred - target) ** 2, (len(pred), -1)), axis=1)
        if self.reduction == 'mean':
            return np.mean(loss)
        else:
            return loss


class CrossEntropyLossNP:
    '''Cross entropy loss that expects probabilities. For use with sklearn
    models.'''
    def __init__(self, reduction='mean'):
        assert reduction in ('none', 'mean')
        self.reduction = reduction

    def __call__(self, pred, target):
        loss = - np.log(pred[np.arange(len(pred)), target])
        if self.reduction == 'mean':
            return np.mean(loss)
        else:
            return loss


class AccuracyNP:
    '''0-1 loss. For use with sklearn models.'''
    def __init__(self):
        pass

    def __call__(self, pred, target):
        '''Calculate 0-1 accuracy.'''
        return np.mean(np.argmax(pred, axis=1) == target)


class SklearnClassifierWrapper:
    '''For sklearn classification models, which require the predict_proba
    function to output probabilities.'''
    def __init__(self, model):
        self.model = model

    def predict(self, x):
        # Output classification probabilities.
        return self.model.predict_proba(x)

