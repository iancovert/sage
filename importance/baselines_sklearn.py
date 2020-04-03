import torch
import sklearn
import numpy as np
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler
from tqdm import tqdm_notebook as tqdm
from models.utils import validate_sklearn, SklearnClassifierWrapper
import importance.utils as utils


def permutation_importance(model,
                           dataset,
                           loss,
                           batch_size,
                           num_permutations=1,
                           bar=False):
    '''
    Calculate feature importance by measuring performance reduction when
    features are replaced with draws from their marginal distribution.

    Args:
      model: sklearn model.
      dataset: PyTorch dataset, such as data.utils.TabularDataset.
      loss: string descriptor of loss function ('mse', 'cross entropy').
      batch_size: number of examples to be processed at once.
      num_permutations: number of sampled values per example.
      bar: whether to display progress bar.
    '''
    # Add wrapper if necessary.
    if isinstance(model, sklearn.base.ClassifierMixin):
        model = SklearnClassifierWrapper(model)

    # Setup.
    input_size = dataset.input_size
    loader = DataLoader(
        dataset, batch_sampler=BatchSampler(
            SequentialSampler(dataset), batch_size=batch_size, drop_last=False))
    loss_fn = utils.get_loss_np(loss, reduction='none')
    scores = []

    # Performance with all features.
    base_loss = validate_sklearn(
        model, loader, utils.get_loss_np(loss, reduction='mean'))

    # For imputing from marginal distribution.
    imputation = utils.MarginalImputation(torch.tensor(dataset.data))

    if bar:
        bar = tqdm(total=num_permutations * input_size * len(dataset))
    for ind in range(input_size):
        # Setup.
        score = 0
        N = 0

        for x, y in loader:
            n = len(x)
            for _ in range(num_permutations):
                # Sample from marginal and make predictions.
                y_hat = model.predict(
                    imputation.impute_ind(x, ind).cpu().data.numpy())

                # Measure loss and compute average.
                loss = np.mean(loss_fn(y_hat, y.cpu().data.numpy()))
                score = (score * N + loss * n) / (N + n)
                N += n
                if bar:
                    bar.update(n)

        scores.append(score)

    return np.stack(scores) - base_loss


def mean_importance(model,
                    dataset,
                    loss,
                    batch_size,
                    bar=False):
    '''
    Calculate feature importance by measuring performance reduction when
    features are imputed with their mean value.

    Args:
      model: sklearn model.
      dataset: PyTorch dataset, such as data.utils.TabularDataset.
      loss: string descriptor of loss function ('mse', 'cross entropy').
      batch_size: number of examples to be processed at once.
      bar: whether to display progress bar.
    '''
    # Add wrapper if necessary.
    if isinstance(model, sklearn.base.ClassifierMixin):
        model = SklearnClassifierWrapper(model)

    # Setup.
    input_size = dataset.input_size
    loader = DataLoader(
        dataset, batch_sampler=BatchSampler(
            SequentialSampler(dataset), batch_size=batch_size, drop_last=False))
    loss_fn = utils.get_loss_np(loss, reduction='none')
    scores = []

    # Performance with all features.
    base_loss = validate_sklearn(
        model, loader, utils.get_loss_np(loss, reduction='mean'))

    # For imputing with mean.
    imputation = utils.ReferenceImputation(
        torch.mean(torch.tensor(dataset.data), dim=0))

    if bar:
        bar = tqdm(total=len(dataset) * input_size)
    for ind in range(input_size):
        # Setup.
        score = 0
        N = 0

        for x, y in loader:
            # Impute with mean and make predictions.
            n = len(x)
            y_hat = model.predict(
                imputation.impute_ind(x, ind).cpu().data.numpy())

            # Measure loss and compute average.
            loss = np.mean(loss_fn(y_hat, y.cpu().data.numpy()))
            score = (score * N + loss * n) / (N + n)
            N += n
            if bar:
                bar.update(n)

        scores.append(score)

    return np.stack(scores) - base_loss

