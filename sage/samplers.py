import numpy as np
from sage import utils
from tqdm.auto import tqdm


def estimate_total(model, xy, batch_size, loss_fn):
    X, Y = xy
    N = 0
    mean_loss = 0
    marginal_pred = 0
    for i in range(np.ceil(len(X) / batch_size).astype(int)):
        x = X[i * batch_size:(i + 1) * batch_size]
        y = Y[i * batch_size:(i + 1) * batch_size]
        n = len(x)
        pred = model(x)
        loss = loss_fn(pred, y)
        marginal_pred = (
            (N * marginal_pred + n * np.mean(pred, axis=0, keepdims=True))
            / (N + n))
        mean_loss = (N * mean_loss + n * np.mean(loss)) / (N + n)
        N += n

    # Mean loss of mean prediction.
    N = 0
    marginal_loss = 0
    for i in range(np.ceil(len(X) / batch_size).astype(int)):
        x = X[i * batch_size:(i + 1) * batch_size]
        y = Y[i * batch_size:(i + 1) * batch_size]
        n = len(x)
        marginal_pred_repeat = marginal_pred.repeat(len(y), 0)
        loss = loss_fn(marginal_pred_repeat, y)
        marginal_loss = (N * marginal_loss + n * np.mean(loss)) / (N + n)
        N += n
    return marginal_loss - mean_loss


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
                 loss):
        self.model = model
        self.imputer = imputer
        self.loss_fn = utils.get_loss(loss, reduction='none')

    def __call__(self,
                 xy,
                 batch_size,
                 n_permutations=None,
                 detect_convergence=None,
                 convergence_threshold=0.01,
                 verbose=False,
                 bar=False):
        '''
        Estimate SAGE values.

        Args:
          xy: tuple of np.ndarrays for input and output.
          batch_size: number of examples to be processed at once. You should use
            as large of a batch size as possible without exceeding available
            memory.
          n_samples: number of permutations. If not specified, samples
            are taken until the estimates converge.
          detect_convergence: whether to detect convergence of SAGE estimates.
          convergence_threshold: confidence interval threshold for determining
            convergence. Represents portion of estimated sum of SAGE values.
          verbose: whether to print progress messages.
          bar: whether to display progress bar.

        Returns: SAGEValues object.
        '''
        X, Y = xy
        N, input_size = X.shape

        # Verify model.
        X, Y = utils.verify_model_data(self.model, X, Y, self.loss_fn,
                                       batch_size * self.imputer.samples)

        # For detecting cnovergence.
        total = estimate_total(
            self.model, xy, batch_size * self.imputer.samples, self.loss_fn)
        if n_permutations is None:
            # Turn convergence detectio on.
            if detect_convergence is None:
                detect_convergence = True
            elif not detect_convergence:
                detect_convergence = True
                print('Turning convergence detection on')

            # Turn bar off.
            if bar:
                bar = False
                print('Turning bar off')

            # Set n_samples to an extremely large number.
            n_permutations = 1e20

        if detect_convergence:
            assert 0 < convergence_threshold < 1

        # Print message explaining parameter choices.
        if verbose:
            print('{} permutations, batch size (batch x samples) = {}'.format(
                n_permutations, batch_size * self.imputer.samples))

        # For updating scores.
        tracker = utils.ImportanceTracker()

        # Permutation sampling.
        n_loops = int(n_permutations / batch_size)
        if bar:
            bar = tqdm(total=n_loops * batch_size * input_size)
        for _ in range(n_loops):
            # Sample data.
            mb = np.random.choice(N, batch_size)
            x = X[mb]
            y = Y[mb]

            # Sample permutations.
            S = np.zeros((batch_size, input_size))
            permutations = np.tile(np.arange(input_size), (batch_size, 1))
            for i in range(batch_size):
                np.random.shuffle(permutations[i])

            # Make prediction with missing features.
            y_hat = self.model(self.imputer(x, S))
            y_hat = np.mean(y_hat.reshape(
                -1, self.imputer.samples, *y_hat.shape[1:]), axis=1)
            prev_loss = self.loss_fn(y_hat, y)

            # Setup.
            arange = np.arange(batch_size)
            scores = np.zeros((batch_size, input_size))

            for i in range(input_size):
                # Add next feature.
                inds = permutations[:, i]
                S[arange, inds] = 1.0

                # Make prediction with missing features.
                y_hat = self.model(self.imputer(x, S))
                y_hat = np.mean(y_hat.reshape(
                    -1, self.imputer.samples, *y_hat.shape[1:]), axis=1)
                loss = self.loss_fn(y_hat, y)

                # Calculate delta sample.
                scores[arange, inds] = prev_loss - loss
                prev_loss = loss
                if bar:
                    bar.update(batch_size)

            # Update tracker.
            tracker.update(scores)

            # Check for convergence.
            conf = np.max(tracker.std)
            if verbose:
                print('Conf = {:.4f}, Total = {:.4f}'.format(conf, total))
            if detect_convergence:
                if (conf / total) < convergence_threshold:
                    if verbose:
                        print('Stopping early')
                    break

        return utils.SAGEValues(tracker.values, tracker.std)


class IteratedSampler:
    '''
    Estimate SAGE values one at a time, by sampling subsets of features.

    Args:
      model: callable prediction model.
      imputer: for imputing held out values.
      loss: loss function ('mse', 'cross entropy').
    '''
    def __init__(self,
                 model,
                 imputer,
                 loss):
        self.model = model
        self.imputer = imputer
        self.loss_fn = utils.get_loss(loss, reduction='none')

    def __call__(self,
                 xy,
                 batch_size,
                 n_samples=None,
                 detect_convergence=False,
                 convergence_threshold=0.01,
                 verbose=False,
                 bar=False):
        '''
        Estimate SAGE values.

        Args:
          xy: tuple of np.ndarrays for input and output.
          batch_size: number of examples to be processed at once. You should use
            as large of a batch size as possible without exceeding available
            memory.
          n_samples: number of samples for each feature. If not specified,
            samples are taken until the estimates converge.
          detect_convergence: whether to detect convergence of SAGE estimates.
          convergence_threshold: confidence interval threshold for determining
            convergence. Represents portion of estimated sum of SAGE values.
          verbose: whether to print progress messages.
          bar: whether to display progress bar.

        Returns: SAGEValues object.
        '''
        X, Y = xy
        N, input_size = X.shape

        # Verify model.
        X, Y = utils.verify_model_data(self.model, X, Y, self.loss_fn,
                                       batch_size * self.imputer.samples)

        # For detecting cnovergence.
        total = estimate_total(
            self.model, xy, batch_size * self.imputer.samples, self.loss_fn)
        if n_samples is None:
            # Turn convergence detectio on.
            if detect_convergence is None:
                detect_convergence = True
            elif not detect_convergence:
                detect_convergence = True
                print('Turning convergence detection on')

            # Turn bar off.
            if bar:
                bar = False
                print('Turning bar off')

            # Set n_samples to an extremely large number.
            n_samples = 1e20

        if detect_convergence:
            assert 0 < convergence_threshold < 1

        if verbose:
            print('{} samples/feat, batch size (batch x samples) = {}'.format(
                n_samples, batch_size * self.imputer.samples))

        # For updating scores.
        tracker_list = []

        # Iterated sampling.
        n_loops = int(n_samples / batch_size)
        if bar:
            bar = tqdm(total=n_loops * batch_size * input_size)
        for ind in range(input_size):
            tracker = utils.ImportanceTracker()
            for _ in range(n_loops):
                # Sample data.
                mb = np.random.choice(N, batch_size)
                x = X[mb]
                y = Y[mb]

                # Sample subset of features.
                S = utils.sample_subset_feature(input_size, batch_size, ind)

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
                if bar:
                    bar.update(batch_size)

                # Check for convergence.
                conf = tracker.std
                if verbose:
                    print('Imp = {:.4f}, Conf = {:.4f}, Total = {:.4f}'.format(
                        tracker.values, conf, total))
                if detect_convergence:
                    if (conf / total) < convergence_threshold:
                        if verbose:
                            print('Stopping feature early')
                        break

            if verbose:
                print('Done with feature {}'.format(ind))
            tracker_list.append(tracker)

        return utils.SAGEValues(
            np.array([tracker.values.item() for tracker in tracker_list]),
            np.array([tracker.std.item() for tracker in tracker_list]))
