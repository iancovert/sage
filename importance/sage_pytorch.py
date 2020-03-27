import torch
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
from tqdm import tqdm_notebook as tqdm
import importance.utils as utils


def estimate_total(model,
                   dataset,
                   batch_size,
                   loss_fn):
    # Estimate expected sum of values.
    device = next(model.parameters()).device
    sequential_loader = DataLoader(dataset, batch_size=batch_size)
    with torch.no_grad():
        N = 0
        mean_loss = 0
        marginal_pred = 0
        for x, y in sequential_loader:
            n = len(x)
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            marginal_pred = (
                (N * marginal_pred + n * torch.mean(pred, dim=0, keepdim=True))
                / (N + n))
            mean_loss = (N * mean_loss + n * torch.mean(loss)) / (N + n)
            N += n

        # Mean loss of mean prediction.
        N = 0
        marginal_loss = 0
        for x, y in sequential_loader:
            n = len(x)
            y = y.to(device)
            marginal_pred_repeat = marginal_pred.repeat(
                *([len(y)] + [1 for _ in range(len(marginal_pred.shape) - 1)]))
            loss = loss_fn(marginal_pred_repeat, y)
            marginal_loss = (N * marginal_loss + n * torch.mean(loss)) / (N + n)
            N += n
    return (marginal_loss - mean_loss).item()


def permutation_sampling(model,
                         dataset,
                         imputation_module,
                         loss,
                         batch_size,
                         n_samples,
                         m_samples,
                         detect_convergence=False,
                         convergence_threshold=0.01,
                         verbose=False,
                         bar=False):
    '''
    Estimates SAGE values by unrolling permutations of feature indices.

    Args:
      model:
      dataset:
      imputation_module:
      loss:
      batch_size:
      n_samples:
      m_samples:
      detect_convergence:
      convergence_threshold:
      verbose:
      bar:
    '''
    # Setup.
    device = next(model.parameters()).device
    input_size = dataset.input_size
    loader = DataLoader(
        dataset, batch_sampler=BatchSampler(
            RandomSampler(dataset, replacement=True,
                          num_samples=n_samples),
            batch_size=batch_size, drop_last=False),
        num_workers=4, pin_memory=True)
    loss_fn = utils.get_loss_pytorch(loss, reduction='none')
    total = estimate_total(model, dataset, batch_size, loss_fn)

    # Print message explaining parameter choices.
    if verbose:
        print('{} samples per feature, minibatch size (batch x m) = {}'.format(
            n_samples, batch_size * m_samples))

    # For updating scores.
    tracker = utils.ImportanceTracker()

    if bar:
        bar = tqdm(total=n_samples * input_size)
    with torch.no_grad():
        for x, y in loader:
            # Move to GPU.
            n = len(x)
            x = x.to(device=device)
            y = y.to(device=device)

            # Sample permutations.
            S = torch.zeros(
                n, input_size, dtype=torch.float32, device=device)
            permutations = torch.arange(input_size).repeat(n, 1)
            for i in range(n):
                permutations.data[i] = (
                    permutations[i, torch.randperm(input_size)])
            S = S.repeat(m_samples, 1)
            permutations = permutations.repeat(m_samples, 1)

            # Make prediction with missing features.
            x = x.repeat(m_samples, 1)
            y_hat = model(imputation_module.impute(x, S))
            y_hat = torch.mean(
                y_hat.reshape(m_samples, -1, *y_hat.shape[1:]), dim=0)
            prev_loss = loss_fn(y_hat, y)

            # Setup.
            arange = torch.arange(n)
            arange_long = torch.arange(n * m_samples)
            scores = torch.zeros(
                n, input_size, dtype=torch.float32, device=device)

            for i in range(input_size):
                # Add next feature.
                inds = permutations[:, i]
                S[arange_long, inds] = 1.0

                # Make prediction with missing features.
                y_hat = model(imputation_module.impute(x, S))
                y_hat = torch.mean(
                    y_hat.reshape(m_samples, -1, *y_hat.shape[1:]), dim=0)
                loss = loss_fn(y_hat, y)

                # Calculate delta sample.
                scores[arange, inds[:n]] = prev_loss - loss
                prev_loss = loss
                if bar:
                    bar.update(n)

            # Update tracker.
            tracker.update(scores.cpu().data.numpy())

            # Check for convergence.
            conf = np.max(tracker.var) ** 0.5
            if verbose:
                print('Conf = {:.4f}, Total = {:.4f}'.format(conf, total))
            if detect_convergence:
                if (conf / total) < convergence_threshold:
                    if verbose:
                        print('Stopping early')
                    break

    return tracker.scores


def iterated_sampling(model,
                      dataset,
                      imputation_module,
                      loss,
                      batch_size,
                      n_samples,
                      m_samples,
                      detect_convergence=False,
                      convergence_threshold=0.01,
                      verbose=False,
                      bar=False):
    '''
    Estimates SAGE values one at a time, by sampling subsets of features.

    Args:
      model:
      dataset:
      imputation_module:
      loss:
      batch_size:
      n_samples:
      m_samples:
      detect_convergence:
      convergence_threshold:
      verbose:
      bar:
    '''
    # Setup.
    device = next(model.parameters()).device
    input_size = dataset.input_size
    loader = DataLoader(
        dataset, batch_sampler=BatchSampler(
            RandomSampler(dataset, replacement=True,
                          num_samples=n_samples),
            batch_size=batch_size, drop_last=False),
        num_workers=4, pin_memory=True)
    loss_fn = utils.get_loss_pytorch(loss, reduction='none')
    total = estimate_total(model, dataset, batch_size, loss_fn)

    # Print message explaining parameter choices.
    if verbose:
        print('{} permutations, minibatch size (batch x m) = {}'.format(
            n_samples, batch_size * m_samples))

    # For updating scores.
    scores = []

    if bar:
        bar = tqdm(total=n_samples * input_size)
    with torch.no_grad():
        for ind in range(input_size):
            tracker = utils.ImportanceTracker()
            for x, y in loader:
                # Move to GPU.
                n = len(x)
                x = x.to(device=device)
                y = y.to(device=device)

                # Sample subset of features.
                S = utils.sample_subset_feature(
                    input_size, n, ind).to(device=device)
                S = S.repeat(m_samples, 1)

                # Loss with feature excluded.
                x = x.repeat(m_samples, 1)
                y_hat = model(imputation_module.impute(x, S))
                y_hat = torch.mean(
                    y_hat.reshape(m_samples, -1, *y_hat.shape[1:]), dim=0)
                loss_discluded = loss_fn(y_hat, y)

                # Loss with feature included.
                S[:, ind] = 1.0
                y_hat = model(
                    imputation_module.impute(x, S))
                y_hat = torch.mean(
                    y_hat.reshape(m_samples, -1, *y_hat.shape[1:]), dim=0)
                loss_included = loss_fn(y_hat, y)

                # Calculate delta sample.
                tracker.update(
                    (loss_discluded - loss_included).cpu().data.numpy())
                if bar:
                    bar.update(n)

                # Check for convergence.
                conf = tracker.var ** 0.5
                if verbose:
                    print('Imp = {:.4f}, Conf = {:.4f}, Total = {:.4f}'.format(
                        tracker.scores, conf, total))
                if detect_convergence:
                    if (conf / total) < convergence_threshold:
                        if verbose:
                            print('Stopping feature early')
                        break

            # Save feature score.
            if verbose:
                print('Done with feature {}'.format(ind))
            scores.append(tracker.scores)

    return np.stack(scores)

