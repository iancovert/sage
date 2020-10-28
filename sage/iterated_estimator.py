import numpy as np
from sage import utils, core
from tqdm.auto import tqdm


def estimate_total(imputer, X, Y, batch_size, loss_fn):
    '''Estimate sum of SAGE values.'''
    N = 0
    mean_loss = 0
    marginal_loss = 0
    num_groups = imputer.num_groups
    for i in range(np.ceil(len(X) / batch_size).astype(int)):
        x = X[i * batch_size:(i + 1) * batch_size]
        y = Y[i * batch_size:(i + 1) * batch_size]
        N += len(x)

        # All features.
        pred = imputer(x, np.ones((len(x), num_groups), dtype=bool))
        loss = loss_fn(pred, y)
        mean_loss += np.sum(loss - mean_loss) / N

        # No features.
        pred = imputer(x, np.zeros((len(x), num_groups), dtype=bool))
        loss = loss_fn(pred, y)
        marginal_loss += np.sum(loss - marginal_loss) / N

    return marginal_loss - mean_loss


def estimate_holdout_importance(imputer, X, Y, batch_size, loss_fn, batches):
    '''Roughly estimate impact of holding out features individually.'''
    N, _ = X.shape
    num_groups = imputer.num_groups
    holdout_importance = np.zeros(num_groups)
    S = np.ones((batch_size, num_groups), dtype=bool)

    # Performance with all features.
    mb = np.random.choice(N, batch_size)
    all_loss = np.mean(loss_fn(imputer(X[mb], S), Y[mb]))

    # Hold out each feature individually.
    for i in range(num_groups):
        S[:, i] = 0
        loss_list = []

        # Multiple minibatches to reduce variance.
        for _ in range(batches):
            mb = np.random.choice(N, batch_size)
            x = X[mb]
            y = Y[mb]
            pred = imputer(x, S)
            loss = loss_fn(pred, y)
            loss_list.append(np.mean(loss))

        # Omit constant term (loss with all features).
        holdout_importance[i] = np.mean(loss_list)
        S[:, i] = 1
    return holdout_importance - all_loss


class IteratedEstimator:
    '''
    Estimate SAGE values one at a time by sampling subsets of features.

    Args:
      imputer: model that accommodates held out features.
      loss: loss function ('mse', 'cross entropy').
    '''
    def __init__(self,
                 imputer,
                 loss='cross entropy'):
        self.imputer = imputer
        self.loss_fn = utils.get_loss(loss, reduction='none')

    def __call__(self,
                 X,
                 Y=None,
                 batch_size=512,
                 detect_convergence=True,
                 thresh=0.025,
                 n_samples=None,
                 optimize_ordering=True,
                 ordering_batches=1,
                 verbose=False,
                 bar=True):
        '''
        Estimate SAGE values.

        Args:
          X: input data.
          Y: target data. If None, model output will be used.
          batch_size: number of examples to be processed in parallel, should be
            set to a large value.
          detect_convergence: whether to stop when approximately converged.
          thresh: threshold for determining convergence
          n_samples: number of samples to take per feature.
          optimize_ordering: whether to guess an ordering of features based on
            importance. May accelerate convergence.
          ordering_batches: number of minibatches while determining ordering.
          verbose: print progress messages.
          bar: display progress bar.

        The default behavior is to detect each feature's convergence based on
        the ratio of its standard deviation to the gap between the largest and
        smallest values. Since neither value is known initially, we begin with
        estimates (upper_val, lower_val) and update them as more features are
        analyzed.

        Returns: Explanation object.
        '''
        # Determine explanation type.
        if Y is not None:
            explanation_type = 'SAGE'
        else:
            explanation_type = 'Shapley Effects'

        # Verify model.
        N, _ = X.shape
        num_features = self.imputer.num_groups
        X, Y = utils.verify_model_data(self.imputer, X, Y, self.loss_fn,
                                       batch_size)

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
            assert 0 < thresh < 1

        # Print message explaining parameter choices.
        if verbose:
            print('Batch size = batch * samples = {}'.format(
                batch_size * self.imputer.samples))

        # For detecting convergence.
        total = estimate_total(self.imputer, X, Y, batch_size, self.loss_fn)
        upper_val = max(total / num_features, 0)
        lower_val = 0

        # Feature ordering.
        if optimize_ordering:
            if verbose:
                print('Determining feature ordering...')
            holdout_importance = estimate_holdout_importance(
                self.imputer, X, Y, batch_size, self.loss_fn, ordering_batches)
            if verbose:
                print('Done')
            # Use np.abs in case there are large negative contributors.
            ordering = list(np.argsort(np.abs(holdout_importance))[::-1])
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
                y_hat = self.imputer(x, S)
                loss_discluded = self.loss_fn(y_hat, y)

                # Loss with feature included.
                S[:, ind] = 1
                y_hat = self.imputer(x, S)
                loss_included = self.loss_fn(y_hat, y)

                # Calculate delta sample.
                tracker.update(loss_discluded - loss_included)
                if bar and (not estimate_convergence):
                    bar.update(batch_size)

                # Calculate progress.
                std = tracker.std
                gap = (
                    max(upper_val, tracker.values.item()) -
                    min(lower_val, tracker.values.item()))
                ratio = std / gap

                # Print progress message.
                if verbose:
                    if detect_convergence:
                        print('StdDev Ratio = {:.4f} '
                              '(Converge at {:.4f})'.format(
                               ratio, thresh))
                    else:
                        print('StdDev Ratio = {:.4f}'.format(ratio))

                # Check for convergence.
                if detect_convergence:
                    if ratio < thresh:
                        if verbose:
                            print('Detected feature convergence')

                        # Skip bar ahead.
                        if bar:
                            bar.n = np.around(
                                bar.total * (i + 1) / num_features, 4)
                            bar.refresh()
                        break

                # Update convergence estimation.
                if bar and estimate_convergence:
                    std_est = ratio * np.sqrt(it + 1)
                    n_est = (std_est / thresh) ** 2
                    bar.n = np.around((i + (it + 1) / n_est) / num_features, 4)
                    bar.refresh()

            if verbose:
                print('Done with feature {}'.format(i))
            tracker_list.append(tracker)

            # Adjust min max value.
            upper_val = max(upper_val, tracker.values.item())
            lower_val = min(lower_val, tracker.values.item())

        if bar:
            bar.close()

        # Extract SAGE values.
        reverse_ordering = [ordering.index(ind) for ind in range(num_features)]
        values = np.array(
            [tracker_list[ind].values.item() for ind in reverse_ordering])
        std = np.array(
            [tracker_list[ind].std.item() for ind in reverse_ordering])

        return core.Explanation(values, std, explanation_type)
