import sys
import numpy as np


def model_conversion(model):
    '''Convert model to callable.'''
    if safe_isinstance(model, 'sklearn.base.ClassifierMixin'):
        return lambda x: model.predict_proba(x)

    elif safe_isinstance(model, 'sklearn.base.RegressorMixin'):
        return lambda x: model.predict(x)

    elif safe_isinstance(model, 'catboost.CatBoostClassifier'):
        return lambda x: model.predict_proba(x)

    elif safe_isinstance(model, 'catboost.CatBoostRegressor'):
        return lambda x: model.predict(x)

    elif safe_isinstance(model, 'lightgbm.basic.Booster'):
        return lambda x: model.predict(x)

    elif safe_isinstance(model, 'xgboost.core.Booster'):
        import xgboost
        return lambda x: model.predict(xgboost.DMatrix(x))

    elif safe_isinstance(model, 'torch.nn.Module'):
        print('Setting up imputer for PyTorch model, assuming that any '
              'necessary output activations are applied properly. If '
              'not, please set up nn.Sequential with nn.Sigmoid or nn.Softmax')

        import torch
        device = next(model.parameters()).device
        return lambda x: model(torch.tensor(
            x, dtype=torch.float32, device=device)).cpu().data.numpy()

    elif callable(model):
        # Assume model is compatible function or callable object.
        return model

    else:
        raise ValueError('model cannot be converted automatically, '
                         'please convert to a lambda function')


def dataset_output(imputer, X, batch_size):
    '''Get model output for entire dataset.'''
    Y = []
    for i in range(int(np.ceil(len(X) / batch_size))):
        x = X[i*batch_size:(i+1)*batch_size]
        pred = imputer(x, np.ones((len(x), imputer.num_groups), dtype=bool))
        Y.append(pred)
    return np.concatenate(Y)


def verify_model_data(imputer, X, Y, loss, batch_size):
    '''Ensure that model and data are set up properly.'''
    check_labels = True
    if Y is None:
        print('Calculating model sensitivity (Shapley Effects, not SAGE)')
        check_labels = False
        Y = dataset_output(imputer, X, batch_size)

        # Fix output shape for classification tasks.
        if isinstance(loss, CrossEntropyLoss):
            if Y.shape == (len(X),):
                Y = Y[:, np.newaxis]
            if Y.shape[1] == 1:
                Y = np.concatenate([1 - Y, Y], axis=1)

    if isinstance(loss, CrossEntropyLoss):
        x = X[:batch_size]
        probs = imputer(x, np.ones((len(x), imputer.num_groups), dtype=bool))

        # Check labels shape.
        if check_labels:
            Y = Y.astype(int)
            if Y.shape == (len(X),):
                # This is the preferred shape.
                pass
            elif Y.shape == (len(X), 1):
                Y = Y[:, 0]
            else:
                raise ValueError('labels shape should be (batch,) or (batch, 1)'
                                 ' for cross entropy loss')

        if (probs.ndim == 1) or (probs.shape[1] == 1):
            # Check label encoding.
            if check_labels:
                unique_labels = np.unique(Y)
                if np.array_equal(unique_labels, np.array([0, 1])):
                    # This is the preferred labeling.
                    pass
                elif np.array_equal(unique_labels, np.array([-1, 1])):
                    # Set -1 to 0.
                    Y = Y.copy()
                    Y[Y == -1] = 0
                else:
                    raise ValueError('labels for binary classification must be '
                                     '[0, 1] or [-1, 1]')

            # Check for valid probability outputs.
            valid_probs = np.all(np.logical_and(probs >= 0, probs <= 1))

        elif probs.ndim == 2:
            # Multiclass output, check for valid probability outputs.
            valid_probs = np.all(np.logical_and(probs >= 0, probs <= 1))
            ones = np.sum(probs, axis=1)
            valid_probs = valid_probs and np.allclose(ones, np.ones(ones.shape))

        else:
            raise ValueError('prediction has too many dimensions')

        if not valid_probs:
            raise ValueError('predictions are not valid probabilities')

    return X, Y


class ImportanceTracker:
    '''For tracking feature importance using a dynamic average.'''
    def __init__(self):
        self.mean = 0
        self.sum_squares = 0
        self.N = 0

    def update(self, scores):
        '''Update mean and sum of squares using Welford's algorithm.'''
        self.N += len(scores)
        diff = scores - self.mean
        self.mean += np.sum(diff, axis=0) / self.N
        diff2 = scores - self.mean
        self.sum_squares += np.sum(diff * diff2, axis=0)

    @property
    def values(self):
        return self.mean

    @property
    def var(self):
        return self.sum_squares / (self.N ** 2)

    @property
    def std(self):
        return self.var ** 0.5


class MSELoss:
    '''MSE loss that sums over non-batch dimensions.'''
    def __init__(self, reduction='mean'):
        assert reduction in ('none', 'mean')
        self.reduction = reduction

    def __call__(self, pred, target):
        # Add dimension if necessary.
        if target.shape[-1] == 1 and len(target.shape) - len(pred.shape) == 1:
            pred = np.expand_dims(pred, -1)
        loss = np.sum(
            np.reshape((pred - target) ** 2, (len(pred), -1)), axis=1)
        if self.reduction == 'mean':
            return np.mean(loss)
        else:
            return loss


class CrossEntropyLoss:
    '''Cross entropy loss that expects probabilities.'''
    def __init__(self, reduction='mean'):
        assert reduction in ('none', 'mean')
        self.reduction = reduction

    def __call__(self, pred, target, eps=1e-12):
        # Clip.
        pred = np.clip(pred, eps, 1 - eps)

        # Add a dimension to prediction probabilities if necessary.
        if pred.ndim == 1:
            pred = pred[:, np.newaxis]
        if pred.shape[1] == 1:
            pred = np.append(1 - pred, pred, axis=1)

        # Calculate loss.
        if target.ndim == 1:
            # Class labels.
            loss = - np.log(pred[np.arange(len(pred)), target])
        elif target.ndim == 2:
            # Probabilistic labels.
            loss = - np.sum(target * np.log(pred), axis=1)
        else:
            raise ValueError('incorrect labels shape for cross entropy loss')

        if self.reduction == 'mean':
            return np.mean(loss)
        else:
            return loss


def get_loss(loss, reduction='mean'):
    '''Get loss function by name.'''
    if loss == 'cross entropy':
        loss_fn = CrossEntropyLoss(reduction=reduction)
    elif loss == 'mse':
        loss_fn = MSELoss(reduction=reduction)
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
    S = np.zeros((n, input_size), dtype=bool)
    choices = list(range(input_size))
    del choices[ind]
    for row in S:
        inds = np.random.choice(
            choices, size=np.random.choice(input_size), replace=False)
        row[inds] = 1
    return S


def safe_isinstance(obj, class_str):
    '''Check isinstance without requiring imports.'''
    if not isinstance(class_str, str):
        return False
    module_name, class_name = class_str.rsplit('.', 1)
    if module_name not in sys.modules:
        return False
    module = sys.modules[module_name]
    class_type = getattr(module, class_name, None)
    if class_type is None:
        return False
    return isinstance(obj, class_type)
