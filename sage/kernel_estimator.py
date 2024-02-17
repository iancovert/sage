import numpy as np
from tqdm.auto import tqdm

from sage import core, utils


def calculate_A(num_features):  # noqa:N802
    """Calculate A parameter's exact form."""
    p_coaccur = (
        np.sum(
            (np.arange(2, num_features) - 1)
            / (num_features - np.arange(2, num_features))
        )
    ) / (
        num_features
        * (num_features - 1)
        * np.sum(
            1
            / (np.arange(1, num_features) * (num_features - np.arange(1, num_features)))
        )
    )
    A = np.eye(num_features) * 0.5 + (1 - np.eye(num_features)) * p_coaccur
    return A


def estimate_constraints(imputer, X, Y, batch_size, loss_fn):
    """
    Estimate loss when no features are included and when all features are
    included. This is used to enforce constraints.
    """
    N = 0
    mean_loss = 0
    marginal_loss = 0
    num_features = imputer.num_groups
    for i in range(np.ceil(len(X) / batch_size).astype(int)):
        x = X[i * batch_size : (i + 1) * batch_size]
        y = Y[i * batch_size : (i + 1) * batch_size]
        N += len(x)

        # All features.
        pred = imputer(x, np.ones((len(x), num_features), dtype=bool))
        loss = loss_fn(pred, y)
        mean_loss += np.sum(loss - mean_loss) / N

        # No features.
        pred = imputer(x, np.zeros((len(x), num_features), dtype=bool))
        loss = loss_fn(pred, y)
        marginal_loss += np.sum(loss - marginal_loss) / N

    return -marginal_loss, -mean_loss


def calculate_result(A, b, total, b_sum_squares, n):
    """Calculate regression coefficients and uncertainty estimates."""
    num_features = A.shape[1]
    A_inv_one = np.linalg.solve(A, np.ones(num_features))
    A_inv_vec = np.linalg.solve(A, b)
    values = A_inv_vec - A_inv_one * (np.sum(A_inv_vec) - total) / np.sum(A_inv_one)

    # Calculate variance.
    try:
        b_sum_squares = 0.5 * (b_sum_squares + b_sum_squares.T)
        b_cov = b_sum_squares / (n**2)
        # TODO this fails in situations where model is invariant to features.
        cholesky = np.linalg.cholesky(b_cov)
        L = np.linalg.solve(A, cholesky) + np.matmul(
            np.outer(A_inv_one, A_inv_one), cholesky
        ) / np.sum(A_inv_one)
        beta_cov = np.matmul(L, L.T)
        var = np.diag(beta_cov)
        std = var**0.5
    except np.linalg.LinAlgError:
        # b_cov likely is not PSD due to insufficient samples.
        std = np.ones(num_features) * np.nan

    return values, std


class KernelEstimator:
    """
    Estimate SAGE values by fitting weighted linear model.

    This is an unbiased estimator designed for stochastic cooperative games,
    described in https://arxiv.org/abs/2012.01536

    Args:
      imputer: model that accommodates held out features.
      loss: loss function ('mse', 'cross entropy', 'zero one').
      random_state: random seed, enables reproducibility.
    """

    def __init__(self, imputer, loss="cross entropy", random_state=None):
        self.imputer = imputer
        self.loss_fn = utils.get_loss(loss, reduction="none")
        self.random_state = random_state

    def __call__(
        self,
        X,
        Y=None,
        batch_size=512,
        detect_convergence=True,
        thresh=0.025,
        n_samples=None,
        verbose=False,
        bar=True,
        check_every=5,
    ):
        """
        Estimate SAGE values by fitting linear regression model.

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
          check_every: number of batches between convergence checks.

        The default behavior is to detect convergence based on the width of the
        SAGE values' confidence intervals. Convergence is defined by the ratio
        of the maximum standard deviation to the gap between the largest and
        smallest values.

        Returns: Explanation object.
        """
        # Set random state.
        self.rng = np.random.default_rng(seed=self.random_state)

        # Determine explanation type.
        if Y is not None:
            explanation_type = "SAGE"
        else:
            explanation_type = "Shapley Effects"

        # Verify model.
        N, _ = X.shape
        num_features = self.imputer.num_groups
        X, Y = utils.verify_model_data(self.imputer, X, Y, self.loss_fn, batch_size)

        # Possibly force convergence detection.
        if n_samples is None:
            n_samples = 1e20
            if not detect_convergence:
                detect_convergence = True
                if verbose:
                    print("Turning convergence detection on")

        if detect_convergence:
            assert 0 < thresh < 1

        # Weighting kernel (probability of each subset size).
        weights = np.arange(1, num_features)
        weights = 1 / (weights * (num_features - weights))
        weights = weights / np.sum(weights)

        # Estimate null and grand coalition values.
        null, grand = estimate_constraints(self.imputer, X, Y, batch_size, self.loss_fn)
        total = grand - null

        # Set up bar.
        n_loops = int(n_samples / batch_size)
        if bar:
            if detect_convergence:
                bar = tqdm(total=1)
            else:
                bar = tqdm(total=n_loops * batch_size)

        # Setup.
        A = calculate_A(num_features)
        n = 0
        b = 0
        b_sum_squares = 0

        # Sample subsets.
        for it in range(n_loops):
            # Sample data.
            mb = self.rng.choice(N, batch_size)
            x = X[mb]
            y = Y[mb]

            # Sample subsets.
            S = np.zeros((batch_size, num_features), dtype=bool)
            num_included = (
                self.rng.choice(num_features - 1, size=batch_size, p=weights) + 1
            )
            for row, num in zip(S, num_included):
                inds = self.rng.choice(num_features, size=num, replace=False)
                row[inds] = 1

            # Calculate loss.
            y_hat = self.imputer(x, S)
            loss = -self.loss_fn(y_hat, y) - null
            b_orig = S.astype(float) * loss[:, np.newaxis]

            # Calculate loss with inverted subset (for variance reduction).
            S = np.logical_not(S)
            y_hat = self.imputer(x, S)
            loss = -self.loss_fn(y_hat, y) - null
            b_inv = S.astype(float) * loss[:, np.newaxis]

            # Welford's algorithm.
            n += batch_size
            b_sample = 0.5 * (b_orig + b_inv)
            b_diff = b_sample - b
            b += np.sum(b_diff, axis=0) / n
            b_diff2 = b_sample - b
            b_sum_squares += np.sum(
                np.matmul(np.expand_dims(b_diff, 2), np.expand_dims(b_diff2, 1)), axis=0
            )

            # Update bar (if not detecting convergence).
            if bar and (not detect_convergence):
                bar.update(batch_size)

            if (it + 1) % check_every == 0:
                # Calculate progress.
                values, std = calculate_result(A, b, total, b_sum_squares, n)
                gap = max(values.max() - values.min(), 1e-12)
                ratio = std.max() / gap

                # Print progress message.
                if verbose:
                    if detect_convergence:
                        print(
                            f"StdDev Ratio = {ratio:.4f} " f"(Converge at {thresh:.4f})"
                        )
                    else:
                        print(f"StdDev Ratio = {ratio:.4f}")

                # Check for convergence.
                if detect_convergence:
                    if ratio < thresh:
                        if verbose:
                            print("Detected convergence")

                        # Skip bar ahead.
                        if bar:
                            bar.n = bar.total
                            bar.refresh()
                        break

                # Update convergence estimation.
                if bar and detect_convergence:
                    N_est = (it + 1) * (ratio / thresh) ** 2
                    bar.n = np.around((it + 1) / N_est, 4)
                    bar.refresh()

        # Calculate SAGE values.
        values, std = calculate_result(A, b, total, b_sum_squares, n)

        return core.Explanation(np.squeeze(values), std, explanation_type)
