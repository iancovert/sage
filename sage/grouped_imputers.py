import warnings

import numpy as np

from sage import utils


def verify_nonoverlapping(groups):
    """Verify that no index is present in more than one group."""
    used_inds = np.concatenate(groups)
    return np.all(np.unique(used_inds, return_counts=True)[1] == 1)


class GroupedImputer:
    """GroupedImputer base class."""

    def __init__(self, model, groups, total, remaining_features):
        self.model = utils.model_conversion(model)

        # Verify that groups are non-overlapping.
        if not verify_nonoverlapping(groups):
            raise ValueError("groups must be non-overlapping")

        # Groups matrix.
        self.groups_mat = np.zeros((len(groups) + 1, total), dtype=bool)
        for i, group in enumerate(groups):
            self.groups_mat[i, group] = 1
        self.groups_mat[-1, :] = 1 - np.sum(self.groups_mat, axis=0)

        # For features that are not specified in any group.
        self.remaining = remaining_features

    def __call__(self, x, S):
        """Calling a GroupedImputer should evaluate the model with the
        specified subset of features."""
        raise NotImplementedError

    def inclusion_matrix(self, S):
        S = np.hstack((S, np.zeros((len(S), 1), dtype=bool)))
        S[:, -1] = self.remaining
        return np.matmul(S, self.groups_mat)


class GroupedDefaultImputer(GroupedImputer):
    """Replace features with default values."""

    def __init__(self, model, values, groups, remaining_features=0):
        super().__init__(model, groups, values.shape[-1], remaining_features)
        if values.ndim == 1:
            values = values[np.newaxis]
        elif values[0] != 1:
            raise ValueError("values shape must be (dim,) or (1, dim)")
        self.values = values
        self.values_repeat = values
        self.num_groups = len(groups)

    def __call__(self, x, S):
        # Prepare x.
        if len(x) != len(self.values_repeat):
            self.values_repeat = self.values.repeat(len(x), 0)

        # Prepare S.
        S = self.inclusion_matrix(S)

        # Replace specified indices.
        x_ = x.copy()
        x_[~S] = self.values_repeat[~S]

        # Make predictions.
        return self.model(x_)


class GroupedMarginalImputer(GroupedImputer):
    """Marginalizing out removed features with their marginal distribution."""

    def __init__(self, model, data, groups, remaining_features=0):
        super().__init__(model, groups, data.shape[1], remaining_features)
        self.data = data
        self.data_repeat = data
        self.samples = len(data)
        self.num_groups = len(groups)

        if len(data) > 1024:
            warnings.warn(
                f"using {len(data)} background samples may lead to slow "
                "runtime, consider using <= 1024",
                RuntimeWarning,
            )

    def __call__(self, x, S):
        # Prepare x.
        n = len(x)
        x = x.repeat(self.samples, 0)

        # Prepare S.
        S = self.inclusion_matrix(S)
        S = S.repeat(self.samples, 0)

        # Prepare samples.
        if len(self.data_repeat) != self.samples * n:
            self.data_repeat = np.tile(self.data, (n, 1))

        # Replace specified indices.
        x_ = x.copy()
        x_[~S] = self.data_repeat[~S]

        # Make predictions.
        pred = self.model(x_)
        pred = pred.reshape(-1, self.samples, *pred.shape[1:])
        return np.mean(pred, axis=1)
