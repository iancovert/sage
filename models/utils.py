import torch
import torch.nn as nn
import numpy as np


def activation_helper(activation):
    '''Get activation function.'''
    if activation == 'sigmoid':
        act = nn.Sigmoid()
    elif activation == 'tanh':
        act = nn.Tanh()
    elif activation == 'relu':
        act = nn.ReLU()
    elif activation == 'softplus':
        act = nn.Softplus()
    elif activation == 'elu':
        act = nn.ELU()
    elif activation == 'softmax':
        act = nn.Softmax(dim=-1)
    elif activation is None:
        act = nn.Identity()
    else:
        raise ValueError('unsupported activation: {}'.format(activation))
    return act


class MSELoss(nn.Module):
    '''MSE loss that always sums over non-batch dimensions.'''
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
    '''Cross entropy loss that expects probabilities.'''
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
    '''0-1 loss.'''
    def __init__(self):
        super(Accuracy, self).__init__()

    def forward(self, pred, target):
        '''Calculate 0-1 accuracy.'''
        return (torch.argmax(pred, dim=1) == target).float().mean()


class NegAccuracy(Accuracy):
    '''0-1 loss, negated for use as validation loss during training.'''
    def __init__(self):
        super(NegAccuracy, self).__init__()

    def forward(self, pred, target):
        '''Negative accuracy, for usage as loss function.'''
        return - super(NegAccuracy, self).forward(pred, target)


class MSELossNP:
    '''MSE loss that always sums over non-batch dimensions.'''
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
    '''Cross entropy loss that expects probabilities.'''
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
    '''0-1 loss.'''
    def __init__(self):
        pass

    def __call__(self, pred, target):
        '''Calculate 0-1 accuracy.'''
        return np.mean(np.argmax(pred, axis=1) == target)


class SklearnClassifierWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, x):
        return self.model.predict_proba(x)


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


def restore_parameters(model, best_model):
    '''Move parameter values from best_model to model.'''
    for param, best_param in zip(model.parameters(), best_model.parameters()):
        param.data = best_param

