# SAGE

**SAGE (Shapley Additive Global importancE)** is a game theoretic approach for understanding black-box machine learning models. It summarizes the importance of each feature based on the predictive power it contributes. Defining feature importance is difficult because of complex feature interactions, like redundancy and complementary behavior, but SAGE accounts for this complexity by considering all subsets of features using the Shapley value from cooperative game theory.

## Install

Please clone our GitHub repository to use the code. The only other packages you'll need are `numpy`, `torch`, `sklearn`, and `tqdm`.

## Usage

SAGE is model-agnostic, but it makes the most sense for tabular datasets where each feature (e.g., age) has a consistent meaning. (Structured data types like images are better understood through individal predictions using *local* interpretability methods, such as [SHAP](https://github.com/slundberg/shap).) Our code currently supports PyTorch and Sklearn models, but we're open to supporting more frameworks.

See `credit.ipynb` for an example using gradient boosting machines (GBM), and see `bike.ipynb` for an example using a PyTorch multi-layer perceptron (MLP).

## Authors

- Ian Covert (<icovert@cs.washington.edu>)
- Scott Lundberg
- Su-In Lee
