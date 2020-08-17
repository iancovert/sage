import numpy as np


class ReferenceImputer:
    '''
    Impute features using reference values.

    Args:
      reference: the reference value for replacing missing features.
    '''
    def __init__(self, reference):
        self.reference = reference
        self.samples = 1

    def __call__(self, x, S):
        x_ = x.copy()
        x_[~S] = self.reference.repeat(len(x), 1)[~S]


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

    def __call__(self, x, S):
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
