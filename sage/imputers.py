import numpy as np
import warnings


class ReferenceImputer:
    '''
    Impute features using reference values.

    Args:
      reference: the reference value for replacing missing features.
    '''
    def __init__(self, reference):
        if reference.ndim == 1:
            reference = reference[np.newaxis]
        elif reference.ndim == 2:
            if reference.shape[0] > 1:
                raise ValueError('expecting one set of reference values, '
                                 'got {}'.format(reference.shape[0]))
        self.reference = reference
        self.reference_repeat = self.reference
        self.samples = 1
        self.num_groups = reference.shape[-1]

    def __call__(self, x, S):
        # Repeat reference.
        if len(self.reference_repeat) != len(x):
            self.reference_repeat = self.reference.repeat(len(x), 0)

        x_ = x.copy()
        x_[~S] = self.reference_repeat[~S]

        return x_


class MarginalImputer:
    '''
    Impute features using a draw from the joint marginal.

    Args:
      data: np.ndarray of size (samples, dimensions) representing the data
        distribution.
      samples: number of samples to draw from marginal distribution.
    '''
    def __init__(self, data, samples):
        self.data = data
        self.samples = samples
        self.N = len(data)
        self.x_addr = None
        self.x_repeat = None
        self.num_groups = data.shape[1]

        if samples > 512:
            warnings.warn('using {} samples may make estimators run '
                          'slowly'.format(samples), RuntimeWarning)

    def __call__(self, x, S):
        # Repeat x.
        if self.x_addr == id(x):
            x = self.x_repeat
        else:
            self.x_addr = id(x)
            x = np.repeat(x, self.samples, 0)
            self.x_repeat = x
        S = np.repeat(S, self.samples, 0)
        samples = self.data[np.random.choice(self.N, len(x), replace=True)]
        x_ = x.copy()
        x_[~S] = samples[~S]
        return x_


class FixedMarginalImputer:
    '''
    Impute features using a fixed background distribution.

    Args:
      data: np.ndarray of size (samples, dimensions) representing the
        background distribution.
    '''
    def __init__(self, data):
        self.data = data
        self.N = len(data)
        self.samples = self.N
        self.x_addr = None
        self.x_repeat = None
        self.data_tiled = data
        self.num_groups = data.shape[1]

        if self.N > 512:
            warnings.warn('using {} background samples may make estimators run '
                          'slowly'.format(self.N), RuntimeWarning)

    def __call__(self, x, S):
        # Repeat x.
        n = len(x)
        if self.x_addr == id(x):
            x = self.x_repeat
        else:
            self.x_addr = id(x)
            x = np.repeat(x, self.N, 0)
            self.x_repeat = x

        # Repeat data.
        if len(self.data_tiled) != (self.N * n):
            self.data_tiled = np.tile(self.data, (n, 1))

        S = np.repeat(S, self.samples, 0)
        x_ = x.copy()
        x_[~S] = self.data_tiled[~S]
        return x_
