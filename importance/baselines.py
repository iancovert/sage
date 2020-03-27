import sklearn
import torch.nn as nn
from importance import baselines_pytorch, baselines_sklearn


def mean_importance(model,
                    dataset,
                    loss,
                    batch_size,
                    bar=True):
    '''
    Calculate feature importance by measuring performance reduction when
    features are imputed with their mean value.

    Args:
      model:
      dataset:
      loss:
      batch_size:
      bar:
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


def permutation_importance(model,
                           dataset,
                           loss,
                           batch_size,
                           num_permutations=1,
                           bar=True):
    '''
    Calculated feature importance by measuring performance reduction when
    features are replaced with draws from their marginal distribution.

    Args:
      model:
      dataset:
      loss:
      batch_size:
      num_permutations:
      bar:
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

