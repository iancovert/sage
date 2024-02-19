import joblib
import numpy as np
from tqdm.auto import tqdm

from sage import core, utils


class PermutationEstimator:
    """
    Estimate SAGE values by unrolling permutations of feature indices.

    Args:
      imputer: model that accommodates held out features.
      loss: loss function ('mse', 'cross entropy', 'zero one').
      n_jobs: number of jobs for parallel processing.
      random_state: random seed, enables reproducibility.
    """

    def __init__(self, imputer, loss="cross entropy", n_jobs=1, random_state=None):
        self.imputer = imputer
        self.loss_fn = utils.get_loss(loss, reduction="none")
        self.random_state = random_state
        self.n_jobs = joblib.effective_n_jobs(n_jobs)
        if n_jobs != 1:
            print(f"PermutationEstimator will use {self.n_jobs} jobs")

    def __call__(
        self,
        X,
        Y=None,
        batch_size=512,
        detect_convergence=True,
        thresh=0.025,
        n_permutations=None,
        min_coalition=0.0,
        max_coalition=1.0,
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
          detect_convergence: whether to stop when approximately converged.
          thresh: threshold for determining convergence.
          n_permutations: number of permutations to unroll.
          min_coalition: minimum coalition size (int or float).
          max_coalition: maximum coalition size (int or float).
          verbose: print progress messages.
          bar: display progress bar.

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

        # Determine min/max coalition sizes.
        if isinstance(min_coalition, float):
            min_coalition = int(min_coalition * num_features)
        if isinstance(max_coalition, float):
            max_coalition = int(max_coalition * num_features)
        assert min_coalition >= 0
        assert max_coalition <= num_features
        assert min_coalition < max_coalition
        if min_coalition > 0 or max_coalition < num_features:
            explanation_type = "Relaxed " + explanation_type

        # Possibly force convergence detection.
        if n_permutations is None:
            n_permutations = 1e20
            if not detect_convergence:
                detect_convergence = True
                if verbose:
                    print("Turning convergence detection on")

        if detect_convergence:
            assert 0 < thresh < 1

        # Set up bar.
        n_loops = int(np.ceil(n_permutations / (batch_size * self.n_jobs)))
        if bar:
            if detect_convergence:
                bar = tqdm(total=1)
            else:
                bar = tqdm(total=n_loops * self.n_jobs * batch_size)

        # Setup.
        tracker = utils.ImportanceTracker()

        for it in range(n_loops):
            # Sample data.
            batches = []
            for _ in range(self.n_jobs):
                idxs = self.rng.choice(N, batch_size)
                batches.append((X[idxs], Y[idxs]))

            # Get results from parallel processing of batches.
            results = joblib.Parallel(n_jobs=self.n_jobs)(
                joblib.delayed(self._process_sample)(
                    x, y, num_features, min_coalition, max_coalition
                )
                for x, y in batches
            )

            for scores, sample_counts in results:
                tracker.update(scores, sample_counts)

            # Calculate progress.
            std = np.max(tracker.std)
            gap = max(tracker.values.max() - tracker.values.min(), 1e-12)
            ratio = std / gap

            # Print progress message.
            if verbose:
                if detect_convergence:
                    print(f"StdDev Ratio = {ratio:.4f} " f"(Converge at {thresh:.4f})")
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

            # Update progress bar.
            if bar and detect_convergence:
                # Update using convergence estimation.
                N_est = (it + 1) * (ratio / thresh) ** 2
                bar.n = np.around((it + 1) / N_est, 4)
                bar.refresh()
            if bar and not detect_convergence:
                # Simply update number of permutations.
                bar.update(self.n_jobs)

        if bar:
            bar.close()

        return core.Explanation(tracker.values, tracker.std, explanation_type)

    def _process_sample(self, x, y, num_features, min_coalition, max_coalition):
        # Setup.
        batch_size = len(x)
        arange = np.arange(batch_size)
        scores = np.zeros((batch_size, num_features))
        S = np.zeros((batch_size, num_features), dtype=bool)
        permutations = np.tile(np.arange(num_features), (batch_size, 1))

        # Sample permutations.
        for i in range(batch_size):
            self.rng.shuffle(permutations[i])

        # Calculate sample counts.
        if min_coalition > 0 or max_coalition < num_features:
            sample_counts = np.zeros(num_features, dtype=int)
            for i in range(batch_size):
                sample_counts[permutations[i, min_coalition:max_coalition]] += 1
        else:
            sample_counts = None

        # Add necessary features to minimum coalition.
        for i in range(min_coalition):
            # Add next feature.
            inds = permutations[:, i]
            S[arange, inds] = 1

        # Make prediction with minimum coalition.
        y_hat = self.imputer(x, S)
        prev_loss = self.loss_fn(y_hat, y)

        # Add all remaining features.
        for i in range(min_coalition, max_coalition):
            # Add next feature.
            inds = permutations[:, i]
            S[arange, inds] = 1

            # Make prediction with missing features.
            y_hat = self.imputer(x, S)
            loss = self.loss_fn(y_hat, y)

            # Calculate delta sample.
            scores[arange, inds] = prev_loss - loss
            prev_loss = loss

        return scores, sample_counts
