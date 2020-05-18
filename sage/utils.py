import numpy as np
import matplotlib.pyplot as plt


class SAGEValues:
    '''For storing and plotting SAGE values.'''
    def __init__(self, values, std):
        self.values = values
        self.std = std

    def plot(self,
             feature_names,
             sort_features=True,
             max_features=np.inf,
             fig=None,
             figsize=(12, 7),
             orientation='horizontal',
             error_bars=True,
             color='forestgreen',
             title='Feature Importance',
             title_size=20,
             y_axis_label_size=14,
             x_axis_label_size=14,
             label_rotation=None,
             tick_size=14):
        '''
        Plot SAGE values.

        Args:
          feature_names: list of feature names.
          sort_features: whether to sort features by their SAGE values.
          max_features: number of features to display.
          fig: matplotlib figure (optional).
          figsize: figure size (if fig is None).
          orientation: horizontal (default) or vertical.
          error_bars: whether to include standard deviation error bars.
          color: bar chart color.
          title: plot title.
          title_size: font size for title.
          y_axis_label_size: font size for y axis label.
          x_axis_label_size: font size for x axix label.
          label_rotation: label rotation (for vertical plots only).
          tick_size: tick sizes (for SAGE value axis only).
        '''
        if fig is None:
            fig = plt.figure(figsize=figsize)
        ax = fig.gca()

        # Sort features if necessary.
        if max_features < len(feature_names):
            sort_features = True
        values = self.values
        std = self.std
        if sort_features:
            argsort = np.argsort(values)[::-1]
            values = values[argsort]
            std = std[argsort]
            feature_names = np.array(feature_names)[argsort]

        # Remove extra features if necessary.
        if max_features < len(feature_names):
            feature_names = list(feature_names[:max_features]) + ['Remaining']
            values = (list(values[:max_features])
                      + [np.sum(values[max_features:])])
            std = (list(std[:max_features])
                   + [np.sum(std[max_features:] ** 2) ** 0.5])

        if orientation == 'horizontal':
            # Bar chart.
            if error_bars:
                ax.barh(np.arange(len(feature_names))[::-1], values,
                        color=color, xerr=std)
            else:
                ax.barh(np.arange(len(feature_names))[::-1], values,
                        color=color)

            # Feature labels.
            if label_rotation is not None:
                raise ValueError('rotation not supported for horizontal charts')
            ax.set_yticks(np.arange(len(feature_names))[::-1])
            ax.set_yticklabels(feature_names, fontsize=y_axis_label_size)

            # Axis labels and ticks.
            ax.set_ylabel('')
            ax.set_xlabel('SAGE values', fontsize=x_axis_label_size)
            ax.tick_params(axis='x', labelsize=tick_size)
        elif orientation == 'vertical':
            # Bar chart.
            if error_bars:
                ax.bar(np.arange(len(feature_names)), values, color=color,
                       yerr=std)
            else:
                ax.bar(np.arange(len(feature_names)), values, color=color)

            # Feature labels.
            if label_rotation is None:
                label_rotation = 90
            if label_rotation < 90:
                ha = 'right'
                rotation_mode = 'anchor'
            else:
                ha = 'center'
                rotation_mode = 'default'
            ax.set_xticks(np.arange(len(feature_names)))
            ax.set_xticklabels(feature_names, rotation=label_rotation, ha=ha,
                               rotation_mode=rotation_mode,
                               fontsize=x_axis_label_size)

            # Axis labels and ticks.
            ax.set_ylabel('SAGE values', fontsize=y_axis_label_size)
            ax.set_xlabel('')
            ax.tick_params(axis='y', labelsize=tick_size)
        else:
            raise ValueError('orientation must be horizontal or vertical')

        ax.set_title(title, fontsize=title_size)
        plt.tight_layout()
        return


class ImportanceTracker:
    '''For tracking feature importance using a dynamic average.'''
    def __init__(self):
        self.first_moment = 0
        self.second_moment = 0
        self.N = 0

    def update(self, scores):
        n = len(scores)
        first_moment = np.mean(scores, axis=0)
        second_moment = np.mean(scores ** 2, axis=0)
        self.first_moment = (
            (self.N * self.first_moment + n * first_moment) / (n + self.N))
        self.second_moment = (
            (self.N * self.second_moment + n * second_moment) / (n + self.N))
        self.N += n

    @property
    def values(self):
        return self.first_moment

    @property
    def var(self):
        return (self.second_moment - self.first_moment ** 2) / self.N

    @property
    def std(self):
        return self.var ** 0.5


class MSELoss:
    '''MSE loss that always sums over non-batch dimensions.'''
    def __init__(self, reduction='mean'):
        assert reduction in ('none', 'mean')
        self.reduction = reduction

    def __call__(self, pred, target):
        # Add dimension to tail of pred, if necessary.
        if target.shape[-1] == 1 and len(target.shape) - len(pred.shape) == 1:
            pred = np.expand_dims(pred, -1)
        loss = np.sum(
            np.reshape((pred - target) ** 2, (len(pred), -1)), axis=1)
        if self.reduction == 'mean':
            return np.mean(loss)
        else:
            return loss


class CrossEntropyLoss:
    # TODO infer whether binary classification based on output size.
    # If (n_samples,) then it's binary. Labels must be (0, 1) or (-1, 1).

    # TODO if (n_samples, k) then it may still be binary, but we don't care.
    # Verify that classes are 0, 1, 2, ..., k.

    # TODO then do this again for accuracy.

    '''Cross entropy loss that expects probabilities.'''
    def __init__(self, reduction='mean'):
        assert reduction in ('none', 'mean')
        self.reduction = reduction

    def __call__(self, pred, target):
        loss = - np.log(pred[np.arange(len(pred)), target])
        if self.reduction == 'mean':
            return np.mean(loss)
        else:
            return loss


class BCELoss:
    '''Binary cross entropy loss that expects probabilities.'''
    def __init__(self, reduction='mean'):
        assert reduction in ('none', 'mean')
        self.reduction = reduction

    def __call__(self, pred, target):
        loss = - target * np.log(pred) - (1 - target) * np.log(1 - pred)
        if self.reduction == 'mean':
            return np.mean(loss)
        else:
            return loss


class Accuracy:
    '''0-1 loss.'''
    def __init__(self, reduction='mean'):
        assert reduction in ('none', 'mean')
        self.reduction = reduction

    def __call__(self, pred, target):
        acc = (np.argmax(pred, axis=1) == target).astype(float)
        if self.reduction == 'mean':
            return np.mean(acc)
        else:
            return acc


class NegAccuracy:
    '''Negative 0-1 loss.'''
    def __init__(self, reduction='mean'):
        assert reduction in ('none', 'mean')
        self.reduction = reduction

    def __call__(self, pred, target):
        neg_acc = - (np.argmax(pred, axis=1) == target).astype(float)
        if self.reduction == 'mean':
            return np.mean(neg_acc)
        else:
            return neg_acc


class BinaryAccuracy:
    '''0-1 loss for binary classifier.'''
    def __init__(self, reduction='mean'):
        assert reduction in ('none', 'mean')
        self.reduction = reduction

    def __call__(self, pred, target):
        acc = ((pred > 0.5) == target).astype(float)
        if self.reduction == 'mean':
            return np.mean(acc)
        else:
            return acc


class NegBinaryAccuracy:
    '''Negative 0-1 loss for binary classifier.'''
    def __init__(self, reduction='mean'):
        assert reduction in ('none', 'mean')
        self.reduction = reduction

    def __call__(self, pred, target):
        neg_acc = - ((pred > 0.5) == target).astype(float)
        if self.reduction == 'mean':
            return np.mean(neg_acc)
        else:
            return neg_acc


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
        return S * x + (1 - S) * self.reference


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
        return S * x + (1 - S) * samples


def get_loss(loss, reduction='mean'):
    '''Get loss function by name.'''
    if loss == 'cross entropy':
        loss_fn = CrossEntropyLoss(reduction=reduction)
    elif loss == 'binary cross entropy':
        loss_fn = BCELoss(reduction=reduction)
    elif loss == 'mse':
        loss_fn = MSELoss(reduction=reduction)
    elif loss == 'accuracy':
        loss_fn = NegAccuracy(reduction=reduction)
    elif loss == 'binary accuracy':
        loss_fn = NegBinaryAccuracy(reduction=reduction)
    else:
        raise ValueError('unsupported loss: {}'.format(loss))
    return loss_fn


def sample_subset_feature(input_size, n, ind):
    '''
    Sample a subset of features where a given feature index must not be
    included. This helper function is used for estimating Shapley values, so
    the subset is sampled by 1) sampling the number of features to be included
    from a uniform distribution, and 2) sampling the features to be included.
    '''
    S = np.zeros((n, input_size), dtype=np.float32)
    choices = list(range(input_size))
    del choices[ind]
    for row in S:
        inds = np.random.choice(
            choices, size=np.random.choice(input_size), replace=False)
        row[inds] = 1
    return S


def verify_model_data(model, x, y, loss, mbsize):
    '''Ensure that model and data are set up properly.'''
    # Verify that model output is compatible with labels.
    if isinstance(loss, CrossEntropyLoss) or isinstance(loss, NegAccuracy):
        assert y.shape == (len(x),)
        probs = model(x[:mbsize])
        classes = probs.shape[1]
        assert classes > 1, 'require multiple outputs for multiclass models'
        if len(np.setdiff1d(np.unique(y), np.arange(classes))) == 0:
            # This is the preffered label encoding.
            pass
        elif len(np.setdiff1d(np.unique(y), [-1, 1])) == 0:
            # Set -1s to 0s.
            y = np.copy(y)
            y[y == -1] = 0
        else:
            raise ValueError('labels for multiclass classification must be '
                             '(0, 1, ..., c)')
    elif isinstance(loss, BCELoss) or isinstance(loss, NegBinaryAccuracy):
        assert y.shape == (len(x),)
        if len(np.setdiff1d(np.unique(y), [0, 1])) == 0:
            # This is the preffered label encoding.
            pass
        elif len(np.setdiff1d(np.unique(y), [-1, 1])) == 0:
            # Set -1s to 0s.
            y = np.copy(y)
            y[y == -1] = 0
        else:
            raise ValueError('labels for binary classification must be (0, 1) '
                             'or (-1, 1)')

    # Verify that outputs are probabilities.
    if isinstance(loss, CrossEntropyLoss):
        probs = model(x[:mbsize])
        ones = np.sum(probs, axis=-1)
        if not np.allclose(ones, np.ones(ones.shape)):
            raise ValueError(
                'outputs must be valid probabilities for cross entropy loss')
    elif isinstance(loss, BCELoss):
        probs = model(x[:mbsize])
        if not np.all(np.logical_and(0 <= probs, probs <= 1)):
            raise ValueError(
                'outputs must be valid probabilities for cross entropy loss')

    return x, y
