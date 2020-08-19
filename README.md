# SAGE

**SAGE (Shapley Additive Global importancE)** is a game-theoretic approach for understanding black-box machine learning models. It summarizes each feature's importance based on the predictive power it contributes, and it accounts for complex feature interactions using the Shapley value.

SAGE is described in detail in [this paper](https://arxiv.org/abs/2004.00668), but if you're new to using Shapley values you might want to start by reading this [blog post](https://iancovert.com/blog/understanding-shap-sage/).

## Install

<!--The easiest way to use the code is to install `sage-importance` with `pip`:

```bash
pip install sage-importance
```

Alternatively, you can clone the repository and install the package using the local `setup.py` file:

```bash
pip install .
```-->

The easiest way to get started is to clone the repository and install the package into your Python environment:

```bash
pip install .
```

## Usage

SAGE is model-agnostic, so you can use it with any kind of machine learning model. All you need to do is set up a data imputer and run a sampler:

```python
import sage

# Get data
x, y = ...
feature_names = ...

# Get model
model = ...

# Set up imputer to represent the data distribution
imputer = sage.MarginalImputer(x, samples=512)

# Set up sampler
sampler = sage.PermutationSampler(model, imputer, 'cross entropy')

# Calculate SAGE values
sage_values = sampler(x, y)
sage_values.plot(feature_names)
```

The result will look something like this.

<p align="center">
  <img width="500" src="https://raw.githubusercontent.com/iancovert/sage/master/docs/bike.svg"/>
</p>

**Convergence.** Convergence is determined automatically based on the size of each value's confidence interval, and a progress bar will display the estimated time until convergence.

**Uncertainty estimation.** Standard deviation confidence intervals are returned for each feature's SAGE value.

**Model conversion.** Our back-end requires that models be converted into a consistent format, and this conversion step is performed automatically for XGBoost, CatBoost, LightGBM, sklearn and PyTorch models. If you're using a different kind of model, you'll need to convert it to a callable function (see [here](https://github.com/iancovert/sage/blob/master/sage/utils.py#L163) for examples).

## Examples

<!-- See [bike.ipynb](https://github.com/iancovert/sage/blob/master/bike.ipynb) for an example using XGBoost, [credit.ipynb](https://github.com/iancovert/sage/blob/master/credit.ipynb) for an example using CatBoost, and [airbnb.ipynb](https://github.com/iancovert/sage/blob/master/airbnb.ipynb) for an example using a PyTorch MLP. -->

- [Bike](https://github.com/iancovert/sage/blob/master/notebooks/bike.ipynb) is a simple example using XGBoost, and [Credit]() is a simple example using CatBoost.
- [Airbnb](https://github.com/iancovert/sage/blob/master/notebooks/airbnb.ipynb) shows an example of calculating SAGE values with grouped features (using a PyTorch MLP).
- [Bank](https://github.com/iancovert/sage/blob/master/notebooks/airbnb.ipynb) shows a model monitoring example that uses SAGE to identify features that hurt the model's performance (using CatBoost).
- [MNIST](https://github.com/iancovert/sage/blob/master/notebooks/mnist.ipynb) shows several strategies to accelerate convergence for datasets with many features (feature grouping, different imputing setups).

## Authors

- Ian Covert (<icovert@cs.washington.edu>)
- Scott Lundberg
- Su-In Lee

## References

Ian Covert, Scott Lundberg, Su-In Lee. "Understanding Global Feature Contributions Through Additive Importance Measures." *arXiv preprint arXiv:2004.00668*, 2020.
