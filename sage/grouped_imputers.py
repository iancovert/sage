import numpy as np
import warnings
from copy import deepcopy


def verify_nonoverlapping(groups):
    '''Verify that no index is present in more than one group.'''
    used_inds = np.concatenate(groups)
    return np.all(np.unique(used_inds, return_counts=True)[1] == 1)


def finalize_groups(groups, input_size, remaining_features):
    '''Determine how to handle remaining features.'''
    assert remaining_features in ('split', 'group', 'ignore')
    remaining = np.setdiff1d(list(range(input_size)), np.concatenate(groups))
    if len(remaining) > 0:
        if remaining_features == 'split':
            for ind in remaining:
                groups.append([ind])
        elif remaining_features == 'group':
            groups.append(remaining_features)
    return groups


class GroupedReferenceImputer:
    '''
    Impute grouped features using reference values.

    Args:
      reference: the reference value for replacing missing features.
      groups: feature groups (list of lists).
      remaining_features: how to handle features that are not group members.
    '''
    def __init__(self, reference, groups, remaining_features='split'):
        if reference.ndim == 1:
            reference = reference[np.newaxis]
        elif reference.ndim == 2:
            if reference.shape[0] > 1:
                raise ValueError('expecting one set of reference values, '
                                 'got {}'.format(reference.shape[0]))
        self.reference = reference
        self.reference_repeat = self.reference
        self.samples = 1

        # Verify that groups are non-overlapping.
        input_size = reference.shape[1]
        groups = deepcopy(groups)
        nonoverlapping = verify_nonoverlapping(groups)
        if not nonoverlapping:
            raise ValueError('groups must be non-overlapping')

        # Handle remaining features.
        groups = finalize_groups(groups, input_size, remaining_features)
        self.groups = groups
        self.num_groups = len(groups)

        # Groups matrix.
        self.groups_mat = np.zeros((len(groups), input_size), dtype=bool)
        for i, group in enumerate(groups):
            self.groups_mat[i, group] = 1

    def __call__(self, x, S):
        # Repeat reference.
        if len(self.reference_repeat) != len(x):
            self.reference_repeat = self.reference.repeat(len(x), 0)

        # Convert groups to feature indices.
        S = np.matmul(S, self.groups_mat)

        x_ = x.copy()
        x_[~S] = self.reference.repeat(len(x), 0)[~S]
        return x_


class GroupedMarginalImputer:
    '''
    Impute grouped features using either a fixed set of background examples, or
    a sampled set of background examples.

    Args:
      data: np.ndarray of size (samples, dimensions) representing the data
        distribution.
      groups: feature groups (list of lists).
      remaining_features: how to handle features that are not group members.
    '''
    def __init__(self, data, groups, samples=None, remaining_features='split'):
        self.data = data
        self.N = len(data)

        if samples is None:
            samples = self.N
        elif not samples < self.N:
            raise ValueError('The specified number of samples ({}) for '
                             'GroupedMarginalImputer should be less than the '
                             'length of the dataset ({}). Either use a smaller '
                             'number of samples, or set samples=None to use '
                             'all examples'.format(samples, self.N))
        self.samples = samples

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

        # Verify that groups are non-overlapping.
        input_size = data.shape[1]
        groups = deepcopy(groups)
        nonoverlapping = verify_nonoverlapping(groups)
        if not nonoverlapping:
            raise ValueError('groups must be non-overlapping')

        # Handle remaining features.
        groups = finalize_groups(groups, input_size, remaining_features)
        self.groups = groups
        self.num_groups = len(groups)

        # Groups matrix.
        self.groups_mat = np.zeros((len(groups), input_size), dtype=bool)
        for i, group in enumerate(groups):
            self.groups_mat[i, group] = 1

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
        if self.N == self.samples:
            # Repeat data.
            if len(self.data_tiled) != (self.samples * n):
                self.data_tiled = np.tile(self.data, (n, 1))
            samples = self.data_tiled
        else:
            # Draw samples from marginal distribution.
            samples = self.data[np.random.choice(self.N, len(x), replace=True)]

        # Convert groups to feature indices.
        S = np.matmul(S, self.groups_mat)

        # Replace specified features.
        S = S.repeat(self.samples, 0)
        x_ = x.copy()
        x_[~S] = samples[~S]
        return x_
