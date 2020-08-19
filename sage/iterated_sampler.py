import numpy as np
from sage import utils, core
from tqdm.auto import tqdm


def estimate_total(model, X, Y, batch_size, loss_fn):
    '''
    Estimate sum of SAGE values.

    This is used to get a worst-case estimate of the maximum
    SAGE value. It assumes that the prediction given no features is
    the mean prediction. This is not true of all imputers, and it will
    overestimate the total value in situations where certain features are
    never held out (which may lead to premature convergence).
    '''
    # Mean loss.
    N = 0
    mean_loss = 0
    mean_pred = 0
    for i in range(np.ceil(len(X) / batch_size).astype(int)):
        x = X[i * batch_size:(i + 1) * batch_size]
        y = Y[i * batch_size:(i + 1) * batch_size]
        n = len(x)
        pred = model(x)
        _mean_loss = np.mean(loss_fn(pred, y))
        _mean_pred = np.mean(pred, axis=0, keepdims=True)
        mean_loss = (N * mean_loss + n * _mean_loss) / (N + n)
        mean_pred = (N * mean_pred + n * _mean_pred) / (N + n)
        N += n

    # Mean loss with no features.
    N = 0
    marginal_loss = 0
    for i in range(np.ceil(len(X) / batch_size).astype(int)):
        y = Y[i * batch_size:(i + 1) * batch_size]
        n = len(x)
        # pred = mean_pred.repeat(n, *[1 for _ in mean_pred.shape[1:]])
        pred = mean_pred.repeat(n, 0)
        _mean_loss = np.mean(loss_fn(pred, y))
        marginal_loss = (N * marginal_loss + n * _mean_loss) / (N + n)
        N += n

    return marginal_loss - mean_loss


def estimate_holdout_importance(model, X, Y, imputer, batch_size, loss_fn,
                                batches, bar):
    '''
    Estimate impact of holding out features individually.

    This provides a rough estimate, but the result is only used to
    determine feature ordering.
    '''
    N, _ = X.shape
    num_features = imputer.num_groups
    holdout_importance = np.zeros(num_features)
    S = np.ones((batch_size, num_features), dtype=bool)

    if bar:
        bar = tqdm(total=num_features)

    # Hold out each feature individually.
    for i in range(num_features):
        S[:, i] = 0
        loss_list = []

        # Multiple minibatches to reduce variance.
        for _ in range(batches):
            mb = np.random.choice(N, batch_size)
            x = X[mb]
            y = Y[mb]
            pred = model(imputer(x, S))
            pred = np.mean(pred.reshape(
                -1, imputer.samples, *pred.shape[1:]), axis=1)
            loss = loss_fn(pred, y)
            loss_list.append(np.mean(loss))

        # Omit constant term (loss with all features).
        holdout_importance[i] = np.mean(loss_list)
        S[:, i] = 1
        if bar:
            bar.update(1)

    if bar:
        bar.close()
    return holdout_importance


class IteratedSampler:
    '''
    Estimate SAGE values one at a time by sampling subsets of features.

    Args:
      model: callable prediction model.
      imputer: for imputing held out values.
      loss: loss function ('mse', 'cross entropy').
    '''
    def __init__(self,
                 model,
                 imputer,
                 loss='cross entropy'):
        self.imputer = imputer
        self.loss_fn = utils.get_loss(loss, reduction='none')
        self.model = utils.model_conversion(model, self.loss_fn)

    def __call__(self,
                 X,
                 Y,
                 batch_size=512,
                 detect_convergence=True,
                 convergence_threshold=0.05,
                 n_samples=None,
                 optimize_ordering=True,
                 ordering_batches=1,
                 verbose=True,
                 bar=True):
        '''
        Estimate SAGE values.

        Args:
          X: input data.
          Y: target data.
          batch_size: number of examples to be processed in parallel, should be
            set to a large value.
          detect_convergence: whether to stop when approximately converged.
          convergence_threshold: threshold for determining convergence,
            represents ratio of standard deviation to max SAGE value.
          n_samples: number of samples to take per feature.
          optimize_ordering: whether to guess an ordering of features based on
            importance. May accelerate convergence.
          ordering_batches: number of minibatches while determining ordering.
          verbose: print progress messages.
          bar: display progress bar.

        The default behavior is to detect each feature's convergence based on
        the ratio of its standard deviation to the maximum value. Since the
        maximum value is unknown, we begin with a worst-case estimate
        (min_max_val) and update this estimate as more features are examined.

        Returns: SAGE object.
        '''
        # Verify model.
        N, _ = X.shape
        num_features = self.imputer.num_groups
        X, Y = utils.verify_model_data(self.model, X, Y, self.loss_fn,
                                       batch_size * self.imputer.samples)

        # For setting up bar.
        estimate_convergence = n_samples is None
        if estimate_convergence and verbose:
            print('Estimating convergence time')

        # Possibly force convergence detection.
        if n_samples is None:
            n_samples = 1e20
            if not detect_convergence:
                detect_convergence = True
                if verbose:
                    print('Turning convergence detection on')

        if detect_convergence:
            assert 0 < convergence_threshold < 1

        # Print message explaining parameter choices.
        if verbose:
            print('Batch size = batch * samples = {}'.format(
                batch_size * self.imputer.samples))

        # For detecting convergence.
        print('Estimating total importance')
        total = estimate_total(
            self.model, X, Y, batch_size * self.imputer.samples, self.loss_fn)
        min_max_val = total / num_features

        # Feature ordering.
        if optimize_ordering:
            if verbose:
                print('Determining feature ordering')
            holdout_importance = estimate_holdout_importance(
                self.model, X, Y, self.imputer, batch_size, self.loss_fn,
                ordering_batches, bar)
            ordering = list(np.argsort(holdout_importance)[::-1])
        else:
            ordering = list(range(num_features))

        # Set up bar.
        n_loops = int(n_samples / batch_size)
        if bar:
            if estimate_convergence:
                bar = tqdm(total=1)
            else:
                bar = tqdm(total=n_loops * batch_size * num_features)

        # Iterated sampling.
        tracker_list = []
        for i, ind in enumerate(ordering):
            tracker = utils.ImportanceTracker()
            for it in range(n_loops):
                # Sample data.
                mb = np.random.choice(N, batch_size)
                x = X[mb]
                y = Y[mb]

                # Sample subset of features.
                S = utils.sample_subset_feature(num_features, batch_size, ind)

                # Loss with feature excluded.
                y_hat = self.model(self.imputer(x, S))
                y_hat = np.mean(y_hat.reshape(
                    -1, self.imputer.samples, *y_hat.shape[1:]), axis=1)
                loss_discluded = self.loss_fn(y_hat, y)

                # Loss with feature included.
                S[:, ind] = 1.0
                y_hat = self.model(self.imputer(x, S))
                y_hat = np.mean(y_hat.reshape(
                    -1, self.imputer.samples, *y_hat.shape[1:]), axis=1)
                loss_included = self.loss_fn(y_hat, y)

                # Calculate delta sample.
                tracker.update(loss_discluded - loss_included)
                if bar and (not estimate_convergence):
                    bar.update(batch_size)

                # Calculate progress.
                std = tracker.std
                max_val = max(min_max_val, np.abs(tracker.values.item()))
                ratio = std / max_val

                # Print progress message.
                if verbose:
                    if detect_convergence:
                        print('StdDev Ratio = {:.4f} '
                              '(Converge at {:.4f})'.format(
                               ratio, convergence_threshold))
                    else:
                        print('StdDev Ratio = {:.4f}'.format(ratio))

                # Check for convergence.
                if detect_convergence:
                    if ratio < convergence_threshold:
                        if verbose:
                            print('Detected feature convergence')

                        # Skip bar ahead.
                        if bar:
                            bar.n = bar.total * (i + 1) / num_features
                            bar.refresh()
                        break

                # Update convergence estimation.
                if bar and estimate_convergence:
                    std_est = ratio * np.sqrt(it + 1)
                    n_est = (std_est / convergence_threshold) ** 2
                    bar.n = np.around((i + (it + 1) / n_est) / num_features, 4)
                    bar.refresh()

            if verbose:
                print('Done with feature {}'.format(i))
            tracker_list.append(tracker)

            # Adjust min max value.
            min_max_val = max(min_max_val, np.abs(tracker.values.item()))

        if bar:
            bar.close()

        # Extract SAGE values.
        reverse_ordering = [ordering.index(ind) for ind in range(num_features)]
        values = np.array(
            [tracker_list[ind].values.item() for ind in reverse_ordering])
        std = np.array(
            [tracker_list[ind].std.item() for ind in reverse_ordering])

        return core.SAGE(values, std)
