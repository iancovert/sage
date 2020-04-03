import sklearn
import torch.nn as nn
from importance import baselines_pytorch, baselines_sklearn


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
      model: prediction model, such as sklearn or PyTorch.
      dataset: PyTorch dataset, such as data.utils.TabularDataset.
      loss: string descriptor of loss function ('mse', 'cross entropy').
      batch_size: number of examples to be processed at once.
      num_permutations: number of sampled values per example.
      bar: whether to display progress bar.
    '''
    if isinstance(model, nn.Module):
        return baselines_pytorch.permutation_importance(model,
                                                        dataset,
                                                        loss,
                                                        batch_size,
                                                        num_permutations,
                                                        bar)
    elif isinstance(model, sklearn.base.BaseEstimator):
        return baselines_sklearn.permutation_importance(model,
                                                        dataset,
                                                        loss,
                                                        batch_size,
                                                        num_permutations,
                                                        bar)
    else:
        raise ValueError('unrecognized model type: {}'.format(type(model)))


def mean_importance(model,
                    dataset,
                    loss,
                    batch_size,
                    bar=False):
    '''
    Calculate feature importance by measuring performance reduction when
    features are imputed with their mean value.

    Args:
      model: prediction model, such as sklearn or PyTorch.
      dataset: PyTorch dataset, such as data.utils.TabularDataset.
      loss: string descriptor of loss function ('mse', 'cross entropy').
      batch_size: number of examples to be processed at once.
      bar: whether to display progress bar.
    '''
    if isinstance(model, nn.Module):
        return baselines_pytorch.mean_importance(model,
                                                 dataset,
                                                 loss,
                                                 batch_size,
                                                 bar)
    elif isinstance(model, sklearn.base.BaseEstimator):
        return baselines_sklearn.mean_importance(model,
                                                 dataset,
                                                 loss,
                                                 batch_size,
                                                 bar)
    else:
        raise ValueError('unrecognized model type: {}'.format(type(model)))

