import torch
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler
from tqdm import tqdm_notebook as tqdm
from models.train_models import validate
import importance.utils as utils


def mean_importance(model,
                    dataset,
                    loss,
                    batch_size,
                    bar=True):
    '''
    Calculate feature importance by measuring performance reduction when
    features are imputed with their mean value.

    Args:
      model:
      dataset:
      loss:
      batch_size:
      bar:
    '''
    # Setup.
    device = next(model.parameters()).device
    input_size = model.input_size
    loader = DataLoader(
        dataset, batch_sampler=BatchSampler(
            SequentialSampler(dataset), batch_size=batch_size, drop_last=False))
    loss_fn = utils.get_loss_pytorch(loss, reduction='none')
    scores = []

    # Performance with all features.
    base_loss = validate(
        model, loader, utils.get_loss_pytorch(loss, reduction='mean')).item()

    # For imputing with mean.
    imputation = utils.ReferenceImputation(
        torch.mean(torch.tensor(dataset.data), dim=0))

    if bar:
        bar = tqdm(total=len(dataset) * input_size)
    with torch.no_grad():
        for ind in range(input_size):
            # Setup.
            score = 0
            N = 0

            for x, y in loader:
                # Move to GPU.
                n = len(x)
                x = x.to(device=device)
                y = y.to(device=device)

                # Impute with mean and make predictions.
                y_hat = model(imputation.impute_ind(x, ind))

                # Measure loss and compute average.
                loss = torch.mean(loss_fn(y_hat, y))
                score = (score * N + loss * n) / (N + n)
                N += n
                if bar:
                    bar.update(n)

            scores.append(score)

    return (torch.stack(scores) - base_loss).cpu().data.numpy()


def permutation_importance(model,
                           dataset,
                           loss,
                           batch_size,
                           num_permutations=1,
                           bar=True):
    '''
    Calculated feature importance by measuring performance reduction when
    features are replaced with draws from their marginal distribution.

    Args:
      model:
      dataset:
      loss:
      batch_size:
      num_permutations:
      bar:
    '''
    # Setup.
    device = next(model.parameters()).device
    input_size = model.input_size
    loader = DataLoader(
        dataset, batch_sampler=BatchSampler(
            SequentialSampler(dataset), batch_size=batch_size, drop_last=False))
    loss_fn = utils.get_loss_pytorch(loss, reduction='none')
    scores = []

    # Performance with all features.
    base_loss = validate(
        model, loader, utils.get_loss_pytorch(loss, reduction='mean')).item()

    # For imputing from marginal distribution.
    imputation = utils.MarginalImputation(torch.tensor(dataset.data))

    if bar:
        bar = tqdm(total=num_permutations * input_size * len(dataset))
    with torch.no_grad():
        for ind in range(input_size):
            # Setup.
            score = 0
            N = 0

            for x, y in loader:
                # Move to GPU.
                n = len(x)
                x = x.to(device=device)
                y = y.to(device=device)

                for _ in range(num_permutations):
                    # Sample from marginal and make predictions.
                    y_hat = model(imputation.impute_ind(x, ind))

                    # Measure loss and compute average.
                    loss = torch.mean(loss_fn(y_hat, y))
                    score = (score * N + loss * n) / (N + n)
                    N += n
                    if bar:
                        bar.update(n)

            scores.append(score)

    return (torch.stack(scores) - base_loss).cpu().data.numpy()

