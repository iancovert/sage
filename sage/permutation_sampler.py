import numpy as np
from sage import utils, core
from tqdm.auto import tqdm


class PermutationSampler:
    '''
    Estimate SAGE values by unrolling permutations of feature indices.

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
                 n_permutations=None,
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
            represents ratio of max standard deviation to max SAGE value.
          n_permutations: number of permutations to unroll.
          verbose: print progress messages.
          bar: display progress bar.

        The default behavior is to detect convergence based on the width of the
        SAGE values' confidence intervals. Convergence is defined by the ratio
        of the maximum standard deviation to the maximum value reaching the
        specified threshold.

        Returns: SAGE object.
        '''
        # Verify model.
        N, _ = X.shape
        num_features = self.imputer.num_groups
        X, Y = utils.verify_model_data(self.model, X, Y, self.loss_fn,
                                       batch_size * self.imputer.samples)

        # For setting up bar.
        estimate_convergence = n_permutations is None
        if estimate_convergence and verbose:
            print('Estimating convergence time')

        # Possibly force convergence detection.
        if n_permutations is None:
            n_permutations = 1e20
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

        # Set up bar.
        n_loops = int(n_permutations / batch_size)
        if bar:
            if estimate_convergence:
                bar = tqdm(total=1)
            else:
                bar = tqdm(total=n_loops * batch_size * num_features)

        # Permutation sampling.
        tracker = utils.ImportanceTracker()
        for it in range(n_loops):
            # Sample data.
            mb = np.random.choice(N, batch_size)
            x = X[mb]
            y = Y[mb]

            # Sample permutations.
            S = np.zeros((batch_size, num_features), dtype=bool)
            permutations = np.tile(np.arange(num_features), (batch_size, 1))
            for i in range(batch_size):
                np.random.shuffle(permutations[i])

            # Make prediction with missing features.
            y_hat = self.model(self.imputer(x, S))
            y_hat = np.mean(y_hat.reshape(
                -1, self.imputer.samples, *y_hat.shape[1:]), axis=1)
            prev_loss = self.loss_fn(y_hat, y)

            # Setup.
            arange = np.arange(batch_size)
            scores = np.zeros((batch_size, num_features))

            for i in range(num_features):
                # Add next feature.
                inds = permutations[:, i]
                S[arange, inds] = 1

                # Make prediction with missing features.
                y_hat = self.model(self.imputer(x, S))
                y_hat = np.mean(y_hat.reshape(
                    -1, self.imputer.samples, *y_hat.shape[1:]), axis=1)
                loss = self.loss_fn(y_hat, y)

                # Calculate delta sample.
                scores[arange, inds] = prev_loss - loss
                prev_loss = loss
                if bar and (not estimate_convergence):
                    bar.update(batch_size)

            # Update tracker.
            tracker.update(scores)

            # Calculate progress.
            std = np.max(tracker.std)
            val = np.max(np.abs(tracker.values))
            ratio = std / val

            # Print progress message.
            if verbose:
                if detect_convergence:
                    print('StdDev Ratio = {:.4f} (Converge at {:.4f})'.format(
                        ratio, convergence_threshold))
                else:
                    print('StdDev Ratio = {:.4f}'.format(ratio))

            # Check for convergence.
            if detect_convergence:
                if ratio < convergence_threshold:
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
                n_est = (std_est / convergence_threshold) ** 2
                bar.n = np.around((it + 1) / n_est, 4)
                bar.refresh()

        if bar:
            bar.close()

        return core.SAGE(tracker.values, tracker.std)
