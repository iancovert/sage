import numpy as np
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
    Impute features using reference values.

    Args:
      reference: the reference value for replacing missing features.
      groups: feature groups (list of lists).
      remaining_features: how to handle features that are not group members.
    '''
    def __init__(self, reference, groups, remaining_features='split'):
        self.reference = reference
        self.samples = 1

        # Verify that groups are non-overlapping.
        input_size = len(reference)
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
        x_ = x.copy()
        S = np.matmul(S, self.groups_mat)
        x_[~S] = self.reference.repeat(len(x), 1)[~S]


class GroupedMarginalImputer:
    '''
    Impute features using a draw from the joint marginal.

    Args:
      data: np.ndarray of size (samples, dimensions) representing the data
        distribution.
      groups: feature groups (list of lists).
      samples: number of samples to draw from marginal distribution.
      remaining_features: how to handle features that are not group members.
    '''
    def __init__(self, data, groups, samples, remaining_features='split'):
        self.data = data
        self.samples = samples
        self.N = len(data)
        self.x_addr = None
        self.x_repeat = None

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
        if self.x_addr == id(x):
            x = self.x_repeat
        else:
            self.x_addr = id(x)
            x = np.repeat(x, self.samples, 0)
            self.x_repeat = x

        S = np.matmul(S, self.groups_mat)
        S = np.repeat(S, self.samples, 0)
        samples = self.data[np.random.choice(self.N, len(x), replace=True)]
        x_ = x.copy()
        x_[~S] = samples[~S]
        return x_
