import torch
import numpy as np
from copy import deepcopy
from models.utils import MSELoss, CrossEntropyLoss
from models.utils import MSELossNP, CrossEntropyLossNP


class ImportanceTracker:
    '''
    To track feature importance values using a dynamically calculated average.
    '''
    def __init__(self):
        self.first_moment = 0
        self.second_moment = 0
        self.N = 0

    def update(self, scores):
        n = len(scores)
        first_moment = np.mean(scores, axis=0)
        second_moment = np.mean(scores ** 2, axis=0)
        self.first_moment = (
            (self.N * self.first_moment + n * first_moment) / (n + self.N))
        self.second_moment = (
            (self.N * self.second_moment + n * second_moment) / (n + self.N))
        self.N += n

    @property
    def scores(self):
        return self.first_moment

    @property
    def var(self):
        return (self.second_moment - self.first_moment ** 2) / self.N


class ReferenceImputation:
    '''
    For imputing sets of features, using a reference value.

    To avoid replicating helper classes and helper functions, this class
    requires PyTorch tensors when calling impute or impute_ind. Functions that
    use these methods should be aware of this, but the PyTorch backend should
    not be exposed to users, who may not use PyTorch models (e.g., sklearn).

    Args:
      reference: the reference value for replacing missing features.
    '''
    def __init__(self, reference):
        if not isinstance(reference, torch.Tensor):
            reference = torch.tensor(reference)
        self.reference = reference.float()

    def impute(self, x, S):
        if self.reference.device != x.device:
            self.reference = self.reference.to(x.device)
        return S * x + (1 - S) * self.reference

    def impute_ind(self, x, ind):
        if self.reference.device != x.device:
            self.reference = self.reference.to(x.device)
        x = deepcopy(x)
        x[:, ind] = self.reference[ind]
        return x


class MarginalImputation:
    '''
    For imputing sets of features, using a draw from the joint marginal.

    To avoid replicating helper classes and helper functions, this class
    requires PyTorch tensors when calling impute or impute_ind. Functions that
    use these methods should be aware of this, but the PyTorch backend should
    not be exposed to users, who may not use PyTorch models (e.g., sklearn).

    Args:
      data: np.ndarray of size (samples, dimensions) representing the data
        distribution.
    '''
    def __init__(self, data):
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data)
        self.data = data.float()
        self.N = len(data)

    def impute(self, x, S):
        samples = self.data[
            np.random.choice(self.N, len(x), replace=True)].to(x.device)
        return S * x + (1 - S) * samples

    def impute_ind(self, x, ind):
        samples = self.data[
            np.random.choice(self.N, len(x), replace=True), ind].to(x.device)
        x = deepcopy(x)
        x[:, ind] = samples
        return x


def get_loss_pytorch(loss, reduction='mean'):
    '''Get loss function by name.'''
    if loss == 'cross entropy':
        loss_fn = CrossEntropyLoss(reduction=reduction)
    elif loss == 'mse':
        loss_fn = MSELoss(reduction=reduction)
    else:
        raise ValueError('unsupported loss: {}'.format(loss))
    return loss_fn


def get_loss_np(loss, reduction='mean'):
    '''Get loss function by name.'''
    if loss == 'cross entropy':
        loss_fn = CrossEntropyLossNP(reduction=reduction)
    elif loss == 'mse':
        loss_fn = MSELossNP(reduction=reduction)
    else:
        raise ValueError('unsupported loss: {}'.format(loss))
    return loss_fn


def sample_subset(input_size, batch_size):
    '''
    Sample a subset of features, with input_size total features. This helper
    function is used for estimating Shapley values, so the subset is sampled by
    1) sampling the number of features from a uniform distribution, 2) sampling
    the features to be included.
    '''
    S = np.zeros((batch_size, input_size), dtype=np.float32)
    for row in S:
        row[:np.random.choice(input_size + 1)] = 1
        np.random.shuffle(row)
    return torch.tensor(S)


def sample_subset_feature(input_size, batch_size, ind):
    '''
    Sample a subset of features, with input_size total features, where the
    feature index ind must be included. This helper function is used for
    estimating Shapley values, so the subset is sampled by 1) sampling the
    number of features to be included from a uniform distribution, 2) sampling
    the features to be included.
    '''
    S = np.zeros((batch_size, input_size), dtype=np.float32)
    choices = list(range(input_size))
    del choices[ind]
    for row in S:
        inds = np.random.choice(
            choices, size=np.random.choice(input_size), replace=False)
        row[inds] = 1.0
    return torch.tensor(S)

