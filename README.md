# SAGE

**SAGE (Shapley Additive Global importancE)** is a game-theoretic approach for understanding black-box machine learning models. It quantifies each feature's importance based on how much predictive power it contributes, and it accounts for complex feature interactions using the Shapley value.

SAGE was introduced in [this paper](https://arxiv.org/abs/2004.00668), but if you're new to using Shapley values you may want to start by reading this [blog post](https://iancovert.com/blog/understanding-shap-sage/).

## Install

The easiest way to get started is to install the `sage-importance` package with `pip`:

```bash
pip install sage-importance
```

Alternatively, you can clone the repository and install the package in your Python environment as follows:

```bash
git clone https://github.com/iancovert/sage.git
cd sage
pip install .
```

## Usage

SAGE is model-agnostic, so you can use it with any kind of machine learning model (linear models, GBMs, neural networks, etc). All you need to do is set up an imputer to handle held out features, and then estimate the Shapley values:

```python
import sage

# Get data
x, y = ...
feature_names = ...

# Get model
model = ...

# Set up an imputer to handle missing features
imputer = sage.MarginalImputer(model, x[:128])

# Set up an estimator
estimator = sage.PermutationEstimator(imputer, 'mse')

# Calculate SAGE values
sage_values = estimator(x, y)
sage_values.plot(feature_names)
```

The result will look like this:

<p align="center">
  <img width="540" src="https://raw.githubusercontent.com/iancovert/sage/master/docs/bike.svg"/>
</p>

Our implementation supports several features to make estimating the Shapley values easier:

- **Uncertainty estimation:** confidence intervals are provided for each feature's importance value.
- **Convergence detection:** convergence is determined based on the size of the confidence intervals, and a progress bar displays the estimated time until convergence.
- **Model conversion:** our back-end requires models to be represented in a consistent format, and this conversion step is performed automatically for XGBoost, CatBoost, LightGBM, sklearn and PyTorch models. If you're using a different kind of model, it needs to be converted to a callable function (see [here](https://github.com/iancovert/sage/blob/master/sage/utils.py#L5) for examples).

## Examples

Check out the following notebooks to get started:

- [Bike](https://github.com/iancovert/sage/blob/master/notebooks/bike.ipynb): a simple example using XGBoost, shows how to calculate SAGE values and Shapley Effects (an alternative explanation when no labels are available)
- [Credit](https://github.com/iancovert/sage/blob/master/notebooks/credit.ipynb): generate explanations using a surrogate model to approximate the conditional distribution (using CatBoost)
- [Airbnb](https://github.com/iancovert/sage/blob/master/notebooks/airbnb.ipynb): calculate SAGE values with grouped features (using a PyTorch MLP)
- [Bank](https://github.com/iancovert/sage/blob/master/notebooks/bank.ipynb): a model monitoring example that uses SAGE to identify features that hurt the model's performance (using CatBoost)
- [MNIST](https://github.com/iancovert/sage/blob/master/notebooks/mnist.ipynb): shows strategies to accelerate convergence for datasets with many features (feature grouping, different imputing setups)
- [Consistency](https://github.com/iancovert/sage/blob/master/notebooks/consistency.ipynb): verifies that our various Shapley value estimators return the same results (see the estimators listed below)
- [Calibration](https://github.com/iancovert/sage/blob/master/notebooks/calibration.ipynb): verifies that SAGE's confidence intervals are representative of the uncertainty across runs
- [Losses](https://github.com/iancovert/sage/blob/master/notebooks/losses.ipynb): shows how SAGE can be used in classification with alternative loss functions.

If you want to replicate the experiments described in our paper, see this separate [repository](https://github.com/iancovert/sage-experiments).

## More details

This repository provides some flexibility in how you generate explanations. You can make several choices when generating explanations.

### 1. Feature removal approach

The original SAGE paper proposes marginalizing out missing features using their conditional distribution. Since this is challenging to implement in practice, several approximations are available. For example, you can:

1. Use default values for missing features (see [MNIST](https://github.com/iancovert/sage/blob/master/notebooks/mnist.ipynb) for an example). This is a fast but low-quality approximation.
2. Sample features from the marginal distribution (see [Bike](https://github.com/iancovert/sage/blob/master/notebooks/bike.ipynb) for an example). This approximation is discussed in the SAGE paper.
3. Train a supervised surrogate model (see [Credit](https://github.com/iancovert/sage/blob/master/notebooks/credit.ipynb) for an example). This approach is described in this [paper](https://arxiv.org/abs/2011.14878), and it can provide a better approximation than the other approaches. However, it requires training an additional model (typically a neural network).
4. Train a model that accommodates missingness. This approach is not shown here, but it's described in this [paper](https://arxiv.org/abs/2011.14878).

### 2. Explanation type

Two types of explanations can be calculated, both based on Shapley values:

1. **SAGE.** This approach quantifies how much each feature improves the model's performance (this is the default).
2. **Shapley Effects.** Described in this [paper](https://epubs.siam.org/doi/pdf/10.1137/130936233?casa_token=fU5qvdv35pkAAAAA:jlQsuRWlPrZ5j3YgaPdOmgOV2-B7FnWB5arog_wj4Sqo4OBTuZsHEgJRPGO7vR1D0UOH8-t9UHU), this explanation method quantifies the model's sensitivity to each feature. Since Shapley Effects is closely related to SAGE (see [here](https://arxiv.org/abs/2011.14878) for details), our implementation generates this type of explanation when labels are not provided. See the [Bike](https://github.com/iancovert/sage/blob/master/notebooks/bike.ipynb) notebook for an example.

### 3. Shapley value estimator

Shapley values are computationally costly to calculate exactly, so we provide several estimation approaches:

1. **Permutation sampling.** This is the approach described in the original paper (see `PermutationEstimator`).
<!--This estimator has an optional argument `min_coalition` that lets you relax the Shapley value's efficiency axiom, often leading to faster importance values with similar properties to SAGE (see [Calibration](https://github.com/iancovert/sage/blob/master/notebooks/calibration.ipynb) for an example).-->
2. **KernelSAGE.** This is a linear regression-based estimator that's similar to KernelSHAP (see `KernelEstimator`). It's described in this [paper](https://arxiv.org/abs/2012.01536), and the [Bank](https://github.com/iancovert/sage/blob/master/notebooks/bank.ipynb) notebook shows an example.
3. **Iterated sampling.** This is a variation on the permutation sampling approach where we calculate Shapley values sequentially for each feature (see `IteratedEstimator`). This enables faster convergence for features with low variance, but it can result in wider confidence intervals.
4. **Sign estimation**. This method estimates SAGE values to a lower precision by focusing only on their sign (i.e., whether they help or hurt performance). It's implemented in `SignEstimator`, and the [Bank](https://github.com/iancovert/sage/blob/master/notebooks/bank.ipynb) notebook shows an example.

The results from each approach should be identical (see [Consistency](https://github.com/iancovert/sage/blob/master/notebooks/consistency.ipynb)), but there may be differences in convergence speed. Permutation sampling is a good approach to start with. KernelSAGE may converge a bit faster, but the uncertainty is spread more evenly among the features rather than being highest for more important features.

### 4. Grouped features

Rather than removing features individually, you can specify groups of features to be removed jointly. This will likely speed up convergence because there are fewer feature subsets. See [Airbnb](https://github.com/iancovert/sage/blob/master/notebooks/airbnb.ipynb) for an example.

## Authors

- Ian Covert (<icovert@cs.washington.edu>)
- Scott Lundberg
- Su-In Lee

## References

Ian Covert, Scott Lundberg, Su-In Lee. "Understanding Global Feature Contributions With Additive Importance Measures." *NeurIPS 2020*

Ian Covert, Scott Lundberg, Su-In Lee. "Explaining by Removing: A Unified Framework for Model Explanation." *JMLR 2021*

Ian Covert, Su-In Lee. "Improving KernelSHAP: Practical Shapley Value Estimation via Linear Regression." *AISTATS 2021*

Art Owen. "Sobol' Indices and Shapley value." *SIAM 2014*
