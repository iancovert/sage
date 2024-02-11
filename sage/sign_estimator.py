import numpy as np
from scipy.stats import norm
from tqdm.auto import tqdm

from sage import core, utils


def estimate_total(imputer, X, Y, batch_size, loss_fn):
    """Estimate sum of SAGE values."""
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

    return marginal_loss - mean_loss


def estimate_holdout_importance(imputer, X, Y, batch_size, loss_fn, batches, rng):
    """Estimate the impact of holding out features individually."""
    N, _ = X.shape
    num_features = imputer.num_groups
    all_loss = 0
    holdout_importance = np.zeros(num_features)
    S = np.ones((batch_size, num_features), dtype=bool)

    # Sample the same batches for all features.
    for it in range(batches):
        # Sample minibatch.
        mb = rng.choice(N, batch_size)
        x = X[mb]
        y = Y[mb]

        # Update loss with all features.
        all_loss += np.sum(loss_fn(imputer(x, S), y) - all_loss) / (it + 1)

        # Loss with features held out.
        for i in range(num_features):
            S[:, i] = 0
            holdout_importance[i] += np.sum(
                loss_fn(imputer(x, S), y) - holdout_importance[i]
            ) / (it + 1)
            S[:, i] = 1

    return holdout_importance - all_loss


class SignEstimator:
    """
    Estimate SAGE values to a lower precision, focusing on the sign. Based on
    the IteratedEstimator strategy of calculating values one at a time.

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
        sign_confidence=0.99,
        narrow_thresh=0.025,
        optimize_ordering=True,
        ordering_batches=1,
        verbose=False,
        bar=True,
    ):
        """
        Estimate SAGE values.

        Args:
          X: input data.
          Y: target data. If None, model output will be used.
          batch_size: number of examples to be processed in parallel, should be
            set to a large value.
          sign_confidence: confidence level on sign.
          narrow_thresh: threshold for detecting that the standard deviation is
            small enough
          optimize_ordering: whether to guess an ordering of features based on
            importance. May accelerate convergence.
          ordering_batches: number of minibatches while determining ordering.
          verbose: print progress messages.
          bar: display progress bar.

        Convergence for each SAGE value is detected when one of two conditions
        holds: (1) the sign is known with high confidence (given by
        sign_confidence), or (2) the standard deviation of the Gaussian
        confidence interval is sufficiently narrow (given by narrow_thresh).

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

        # Verify thresholds.
        assert 0 < narrow_thresh < 1
        assert 0.9 <= sign_confidence < 1
        sign_thresh = 1 / norm.ppf(sign_confidence)

        # For detecting convergence.
        total = estimate_total(self.imputer, X, Y, batch_size, self.loss_fn)
        upper_val = max(total / num_features, 0)
        lower_val = min(total / num_features, 0)

        # Feature ordering.
        if optimize_ordering:
            if verbose:
                print("Determining feature ordering...")
            holdout_importance = estimate_holdout_importance(
                self.imputer, X, Y, batch_size, self.loss_fn, ordering_batches, self.rng
            )
            if verbose:
                print("Done")
            # Use np.abs in case there are large negative contributors.
            ordering = list(np.argsort(np.abs(holdout_importance))[::-1])
        else:
            ordering = list(range(num_features))

        # Set up bar.
        if bar:
            bar = tqdm(total=1)

        # Iterated sampling.
        tracker_list = []
        for i, ind in enumerate(ordering):
            tracker = utils.ImportanceTracker()
            it = 0
            converged = False
            while not converged:
                # Sample data.
                mb = self.rng.choice(N, batch_size)
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

                # Calculate progress.
                val = tracker.values.item()
                std = tracker.std.item()
                gap = max(max(upper_val, val) - min(lower_val, val), 1e-12)
                converged_sign = (std / max(np.abs(val), 1e-12)) < sign_thresh
                converged_narrow = (std / gap) < narrow_thresh

                # Print progress message.
                if verbose:
                    print(
                        "Sign Ratio = {:.4f} (Converge at {:.4f}), "
                        "Narrow Ratio = {:.4f} (Converge at {:.4f})".format(
                            std / np.abs(val), sign_thresh, std / gap, narrow_thresh
                        )
                    )

                # Check for convergence.
                converged = converged_sign or converged_narrow
                if converged:
                    if verbose:
                        print("Detected feature convergence")

                    # Skip bar ahead.
                    if bar:
                        bar.n = np.around(bar.total * (i + 1) / num_features, 4)
                        bar.refresh()

                # Update convergence estimation.
                elif bar:
                    N_sign = (it + 1) * ((std / np.abs(val)) / sign_thresh) ** 2
                    N_narrow = (it + 1) * ((std / gap) / narrow_thresh) ** 2
                    N_est = min(N_sign, N_narrow)
                    bar.n = np.around((i + (it + 1) / N_est) / num_features, 4)
                    bar.refresh()

                # Increment iteration variable.
                it += 1

            if verbose:
                print(f"Done with feature {i}")
            tracker_list.append(tracker)

            # Adjust min max value.
            upper_val = max(upper_val, tracker.values.item())
            lower_val = min(lower_val, tracker.values.item())

        if bar:
            bar.close()

        # Extract SAGE values.
        reverse_ordering = [ordering.index(ind) for ind in range(num_features)]
        values = np.array([tracker_list[ind].values.item() for ind in reverse_ordering])
        std = np.array([tracker_list[ind].std.item() for ind in reverse_ordering])

        return core.Explanation(values, std, explanation_type)
