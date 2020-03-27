import sklearn
import torch.nn as nn
from importance import sage_pytorch, sage_sklearn


def permutation_sampling(model,
                         dataset,
                         imputation_module,
                         loss,
                         batch_size,
                         n_samples,
                         m_samples,
                         detect_convergence=False,
                         convergence_threshold=0.01,
                         verbose=False,
                         bar=False):
    '''
    Estimates SAGE values by unrolling permutations of feature indices.

    Args:
      model:
      dataset:
      imputation_module:
      loss:
      batch_size:
      n_samples:
      m_samples:
      detect_convergence:
      convergence_threshold:
      verbose:
      bar:
    '''
    if isinstance(model, nn.Module):
        return sage_pytorch.permutation_sampling(model,
                                                 dataset,
                                                 imputation_module,
                                                 loss,
                                                 batch_size,
                                                 n_samples,
                                                 m_samples,
                                                 detect_convergence,
                                                 convergence_threshold,
                                                 verbose,
                                                 bar)
    elif isinstance(model, sklearn.base.BaseEstimator):
        return sage_sklearn.permutation_sampling(model,
                                                 dataset,
                                                 imputation_module,
                                                 loss,
                                                 batch_size,
                                                 n_samples,
                                                 m_samples,
                                                 detect_convergence,
                                                 convergence_threshold,
                                                 verbose,
                                                 bar)
    else:
        raise ValueError('unrecognized model type: {}'.format(type(model)))


def iterated_sampling(model,
                      dataset,
                      imputation_module,
                      loss,
                      batch_size,
                      n_samples,
                      m_samples,
                      detect_convergence=False,
                      convergence_threshold=0.01,
                      verbose=False,
                      bar=False):
    '''
    Estimates SAGE values one at a time, by sampling subsets of features.

    Args:
      model:
      dataset:
      imputation_module:
      loss:
      batch_size:
      n_samples:
      m_samples:
      detect_convergence:
      convergence_threshold:
      verbose:
      bar:
    '''
    if isinstance(model, nn.Module):
        return sage_pytorch.iterated_sampling(model,
                                              dataset,
                                              imputation_module,
                                              loss,
                                              batch_size,
                                              n_samples,
                                              m_samples,
                                              detect_convergence,
                                              convergence_threshold,
                                              verbose,
                                              bar)
    elif isinstance(model, sklearn.base.BaseEstimator):
        return sage_sklearn.iterated_sampling(model,
                                              dataset,
                                              imputation_module,
                                              loss,
                                              batch_size,
                                              n_samples,
                                              m_samples,
                                              detect_convergence,
                                              convergence_threshold,
                                              verbose,
                                              bar)
    else:
        raise ValueError('unrecognized model type: {}'.format(type(model)))

