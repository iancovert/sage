import sys
import numpy as np


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

        # Add a dimension if necessary.
        if pred.ndim == 1:
            pred = pred[:, np.newaxis]
        if pred.shape[1] == 1:
            pred = np.append(1 - pred, pred, axis=1)

        # Calculate loss.
        loss = - np.log(pred[np.arange(len(pred)), target])
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


def verify_model_data(model, X, Y, loss, mbsize):
    '''Ensure that model and data are set up properly.'''
    if isinstance(loss, CrossEntropyLoss):
        probs = model(X[:mbsize])
        Y = Y.astype(int)

        if (probs.ndim == 1) or (probs.shape[1] == 1):
            # Single probabilities.
            if Y.shape == (len(X),):
                pass
            elif Y.shape == (len(X), 1):
                Y = Y[:, 0]
            else:
                raise ValueError('labels shape is incorrect')

            valid_probs = np.all(np.logical_and(probs >= 0, probs <= 1))
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

        elif probs.ndim == 2:
            # Multiclass output.
            assert Y.shape == (len(X),)
            ones = np.sum(probs, axis=1)
            valid_probs = np.allclose(ones, np.ones(ones.shape))

        else:
            raise ValueError('predictions array has too many dimensions')

        if not valid_probs:
            raise ValueError('outputs must be valid probabilities')

    return X, Y


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


def model_conversion(model, loss_fn):
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
        import torch
        device = next(model.parameters()).device

        if isinstance(loss_fn, CrossEntropyLoss):
            print('Converting PyTorch classifier, outputs are assumed '
                  'to be logits')

            def f(x):
                x = torch.tensor(x, dtype=torch.float32, device=device)
                out = model(x)
                if out.dim() == 1:
                    out = out.sigmoid()
                elif out.dim() == 2 and out.shape[1] == 1:
                    out = out.sigmoid()
                elif out.dim() == 2:
                    out = out.softmax(dim=1)
                else:
                    raise ValueError('predictions are not valid shape')
                return out.cpu().data.numpy()
        else:
            def f(x):
                x = torch.tensor(x, dtype=torch.float32, device=device)
                out = model(x)
                return out.cpu().data.numpy()

        return f

    elif callable(model):
        # Assume model is compatible function or callable object.
        return model

    else:
        raise ValueError('model cannot be converted automatically, '
                         'sorry! Please convert to a lambda function')
