import numpy as np
from sage import utils, core
from tqdm.auto import tqdm


def estimate_constraints(imputer, X, Y, batch_size, loss_fn):
    '''
    Estimate the loss when no features are included, and when all features
    are included. This is used to ensure that the constraints are set properly.
    '''
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

    return - marginal_loss, - mean_loss


def calculate_result(A, b, v0, v1, b_sum_squares, n):
    '''Calculate regression coefficients and uncertainty estimates.'''
    num_features = A.shape[1]
    A_inv_one = np.linalg.solve(A, np.ones(num_features))
    A_inv_vec = np.linalg.solve(A, b)
    values = (
        A_inv_vec -
        A_inv_one * (np.sum(A_inv_vec) - v1 + v0) / np.sum(A_inv_one))

    # Calculate variance.
    b_sum_squares = 0.5 * (b_sum_squares + b_sum_squares.T)
    b_cov = b_sum_squares / (n ** 2)
    cholesky = np.linalg.cholesky(b_cov)
    L = (
        np.linalg.solve(A, cholesky) +
        np.matmul(np.outer(A_inv_one, A_inv_one), cholesky) / np.sum(A_inv_one))
    beta_cov = np.matmul(L, L.T)
    var = np.diag(beta_cov)
    std = var ** 0.5

    return values, std


class KernelEstimator:
    '''
    Estimate SAGE values by fitting weighted linear model.

    Args:
      imputer: model that accommodates held out features.
      loss: loss function ('mse', 'cross entropy').
    '''
    def __init__(self, imputer, loss):
        self.imputer = imputer
        self.loss_fn = utils.get_loss(loss, reduction='none')

    def __call__(self,
                 X,
                 Y=None,
                 batch_size=512,
                 detect_convergence=True,
                 thresh=0.01,
                 n_samples=None,
                 verbose=False,
                 bar=True,
                 check_every=5):
        '''
        Estimate SAGE values by fitting regression model (like KernelSHAP).

        Args:
          X: input data.
          Y: target data. If None, model output will be used.
          batch_size: number of examples to be processed in parallel, should be
            set to a large value.
          detect_convergence: whether to stop when approximately converged.
          thresh: threshold for determining convergence.
          n_samples: number of permutations to unroll.
          verbose: print progress messages.
          bar: display progress bar.
          check_every: number of batches between progress/convergence checks.

        The default behavior is to detect convergence based on the width of the
        SAGE values' confidence intervals. Convergence is defined by the ratio
        of the maximum standard deviation to the gap between the largest and
        smallest values.

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
        X, Y = utils.verify_model_data(
            self.imputer, X, Y, self.loss_fn, batch_size)

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

        # Weighting kernel (probability of each subset size).
        weights = np.arange(1, num_features)
        weights = 1 / (weights * (num_features - weights))
        weights = weights / np.sum(weights)

        # Estimate v({}) and v(D) for constraints.
        v0, v1 = estimate_constraints(
            self.imputer, X, Y, batch_size, self.loss_fn)

        # Exact form for A.
        p_coaccur = (
            (np.sum((np.arange(2, num_features) - 1) / (num_features - np.arange(2, num_features)))) /
            (num_features * (num_features - 1) *
             np.sum(1 / (np.arange(1, num_features) * (num_features - np.arange(1, num_features))))))
        A = np.eye(num_features) * 0.5 + (1 - np.eye(num_features)) * p_coaccur

        # Set up bar.
        n_loops = int(n_samples / batch_size)
        if bar:
            if estimate_convergence:
                bar = tqdm(total=1)
            else:
                bar = tqdm(total=n_loops * batch_size)

        # Setup.
        n = 0
        b = 0
        b_sum_squares = 0

        # Sample subsets.
        for it in range(n_loops):
            # Sample data.
            mb = np.random.choice(N, batch_size)
            x = X[mb]
            y = Y[mb]

            # Sample subsets.
            S = np.zeros((batch_size, num_features), dtype=bool)
            num_included = np.random.choice(num_features - 1, size=batch_size,
                                            p=weights) + 1
            for row, num in zip(S, num_included):
                inds = np.random.choice(num_features, size=num, replace=False)
                row[inds] = 1

            # Make predictions.
            y_hat = self.imputer(x, S)
            loss = - self.loss_fn(y_hat, y) - v0
            b_temp1 = S.astype(float) * loss[:, np.newaxis]

            # Invert subset for variance reduction.
            S = np.logical_not(S)

            # Make predictions.
            y_hat = self.imputer(x, S)
            loss = - self.loss_fn(y_hat, y) - v0
            b_temp2 = S.astype(float) * loss[:, np.newaxis]

            # Covariance estimate (Welford's algorithm).
            n += batch_size
            b_temp = 0.5 * (b_temp1 + b_temp2)
            b_diff = b_temp - b
            b += np.sum(b_diff, axis=0) / n
            b_diff2 = b_temp - b
            b_sum_squares += np.sum(
                np.matmul(np.expand_dims(b_diff, 2),
                          np.expand_dims(b_diff2, 1)), axis=0)
            if bar and (not estimate_convergence):
                bar.update(batch_size)

            if (it + 1) % check_every == 0:
                # Calculate progress.
                values, std = calculate_result(
                    A, b, v0, v1, b_sum_squares, n)
                gap = values.max() - values.min()
                ratio = np.max(std) / gap

                # Print progress message.
                if verbose:
                    if detect_convergence:
                        print('StdDev Ratio = {:.4f} (Converge at {:.4f})'.format(
                            ratio, thresh))
                    else:
                        print('StdDev Ratio = {:.4f}'.format(ratio))

                # Check for convergence.
                if detect_convergence:
                    if ratio < thresh:
                        if verbose:
                            print('Detected convergence')

                        # Skip bar ahead.
                        if bar:
                            bar.n = bar.total
                            bar.refresh()
                        break

                # Update convergence estimation.
                if bar and estimate_convergence:
                    std_est = ratio * np.sqrt(it + 1)
                    n_est = (std_est / thresh) ** 2
                    bar.n = np.around((it + 1) / n_est, 4)
                    bar.refresh()

        # Calculate SAGE values.
        values, std = calculate_result(A, b, v0, v1, b_sum_squares, n)

        return core.Explanation(np.squeeze(values), std, explanation_type)
