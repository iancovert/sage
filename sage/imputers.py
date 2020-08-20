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
    Impute features using either a fixed set of background examples, or a
    sampled set of background examples.

    Args:
      data: np.ndarray of data points representing the background distribution.
      samples: number of samples to draw (if None, then all samples are used.)
    '''
    def __init__(self, data, samples=None):
        self.data = data
        self.N = len(data)

        if samples is None:
            samples = self.N
        elif not samples < self.N:
            raise ValueError('The specified number of samples ({}) for '
                             'MarginalImputer should be less than the length '
                             'of the dataset ({}). Either use a smaller number '
                             'of samples, or set samples=None to use all '
                             'examples'.format(samples, self.N))
        self.samples = samples
        self.num_groups = data.shape[1]

        # For saving time during imputation.
        self.x_addr = None
        self.x_repeat = None
        if samples == self.N:
            self.data_tiled = data

        # Check if there are too many samples.
        if samples > 512:
            warnings.warn('using {} background samples will make estimators '
                          'run slowly, recommendation is to use <= 512'.format(
                            samples), RuntimeWarning)

    def __call__(self, x, S):
        # Repeat x.
        n = len(x)
        if self.x_addr == id(x):
            x = self.x_repeat
        else:
            self.x_addr = id(x)
            x = x.repeat(self.samples, 0)
            self.x_repeat = x

        # Prepare background samples.
        if self.samples == self.N:
            # Repeat fixed background samples.
            if len(self.data_tiled) != (self.samples * n):
                self.data_tiled = np.tile(self.data, (n, 1))
            samples = self.data_tiled
        else:
            # Draw samples from the marginal distribution.
            samples = self.data[np.random.choice(self.N, len(x), replace=True)]

        # Replace specified features.
        S = S.repeat(self.samples, 0)
        x_ = x.copy()
        x_[~S] = samples[~S]
        return x_
