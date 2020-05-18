# SAGE

**SAGE (Shapley Additive Global importancE)** is a game theoretic approach for understanding black-box machine learning models. It summarizes the importance of each feature based on the predictive power it contributes, and accounts for complex feature interactions by using the Shapley value from cooperative game theory.

SAGE is described in detail in [this paper](https://arxiv.org/abs/2004.00668).

## Usage

SAGE is model-agnostic, so you can use it with any kind of machine learning model. All you need to do is wrap the model in a function to make it callable, set up a data imputer, and run the sampling algorithm:

```python
import sage

# Get data
x, y = ...

# Get model
model = ...

# For representing data distribution
imputer = sage.utils.MarginalImputer(x, samples=512)

# Set up sampling object
sampler = sage.PermutationSampler(
    model,
    imputer,
    'cross entropy')

# Calculate SAGE values
sage_values = sampler(
    (x, y),
    batch_size=256,
    n_permutations=8096,
    bar=True)
```

See [credit.ipynb](https://github.com/icc2115/sage/blob/master/credit.ipynb) for an example using gradient boosting machines (GBM), and [bike.ipynb](https://github.com/icc2115/sage/blob/master/bike.ipynb) for an example using a PyTorch multi-layer perceptron (MLP).

## Install

Please clone our GitHub repository to use the code. The only packages you'll need are `numpy`, `matplotlib` and `tqdm` (plus the packages you need for your machine learning models).

## Authors

- Ian Covert (<icovert@cs.washington.edu>)
- Scott Lundberg
- Su-In Lee

## References

Ian Covert, Scott Lundberg, Su-In Lee. "Understanding Global Feature Contributions Through Additive Importance Measures." *arXiv preprint arXiv:2004.00668*, 2020.
