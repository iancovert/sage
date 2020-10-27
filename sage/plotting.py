import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def plot(explanation,
         feature_names=None,
         sort_features=True,
         max_features=np.inf,
         orientation='horizontal',
         error_bars=True,
         confidence_level=0.95,
         capsize=5,
         color='tab:green',
         title='Feature Importance',
         title_size=20,
         tick_size=16,
         tick_rotation=None,
         label_size=16,
         figsize=(10, 7),
         return_fig=False):
    '''
    Plot SAGE values.

    Args:
      explanation: Explanation object.
      feature_names: list of feature names.
      sort_features: whether to sort features by their values.
      max_features: number of features to display.
      orientation: horizontal (default) or vertical.
      error_bars: whether to include standard deviation error bars.
      confidence_level: confidence interval coverage (e.g., 95%).
      capsize: error bar cap width.
      color: bar chart color.
      title: plot title.
      title_size: font size for title.
      tick_size: font size for feature names and numerical values.
      tick_rotation: tick rotation for feature names (vertical plots only).
      label_size: font size for label.
      figsize: figure size (if fig is None).
      return_fig: whether to return matplotlib figure object.
    '''
    # Default feature names.
    if feature_names is None:
        feature_names = ['Feature {}'.format(i) for i in
                         range(len(explanation.values))]

    # Sort features if necessary.
    if len(feature_names) > max_features:
        sort_features = True

    # Perform sorting.
    values = explanation.values
    std = explanation.std
    if sort_features:
        argsort = np.argsort(values)[::-1]
        values = values[argsort]
        std = std[argsort]
        feature_names = np.array(feature_names)[argsort]

    # Remove extra features if necessary.
    if len(feature_names) > max_features:
        feature_names = (list(feature_names[:max_features])
                         + ['Remaining Features'])
        values = (list(values[:max_features])
                  + [np.sum(values[max_features:])])
        std = (list(std[:max_features])
               + [np.sum(std[max_features:] ** 2) ** 0.5])

    # Warn if too many features.
    if len(feature_names) > 50:
        warnings.warn('Plotting {} features may make figure too crowded, '
                      'consider using max_features'.format(
                        len(feature_names)), Warning)

    # Discard std if necessary.
    if not error_bars:
        std = None
    else:
        assert 0 < confidence_level < 1
        std = std * norm.ppf(0.5 + confidence_level / 2)

    # Make plot.
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()

    if orientation == 'horizontal':
        # Bar chart.
        ax.barh(np.arange(len(feature_names))[::-1], values,
                color=color, xerr=std, capsize=capsize)

        # Feature labels.
        if tick_rotation is not None:
            raise ValueError('rotation not supported for horizontal charts')
        ax.set_yticks(np.arange(len(feature_names))[::-1])
        ax.set_yticklabels(feature_names, fontsize=label_size)

        # Axis labels and ticks.
        ax.set_ylabel('')
        ax.set_xlabel('{} value'.format(explanation.explanation_type),
                      fontsize=label_size)
        ax.tick_params(axis='x', labelsize=tick_size)

    elif orientation == 'vertical':
        # Bar chart.
        ax.bar(np.arange(len(feature_names)), values, color=color,
               yerr=std, capsize=capsize)

        # Feature labels.
        if tick_rotation is None:
            tick_rotation = 45
        if tick_rotation < 90:
            ha = 'right'
            rotation_mode = 'anchor'
        else:
            ha = 'center'
            rotation_mode = 'default'
        ax.set_xticks(np.arange(len(feature_names)))
        ax.set_xticklabels(feature_names, rotation=tick_rotation, ha=ha,
                           rotation_mode=rotation_mode,
                           fontsize=label_size)

        # Axis labels and ticks.
        ax.set_ylabel('{} value'.format(explanation.explanation_type),
                      fontsize=label_size)
        ax.set_xlabel('')
        ax.tick_params(axis='y', labelsize=tick_size)

    else:
        raise ValueError('orientation must be horizontal or vertical')

    # Remove spines.
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_title(title, fontsize=title_size)
    plt.tight_layout()

    if return_fig:
        return fig
    else:
        return


def comparison_plot(comparison_explanations,
                    comparison_names=None,
                    feature_names=None,
                    sort_features=True,
                    max_features=np.inf,
                    orientation='vertical',
                    error_bars=True,
                    confidence_level=0.95,
                    capsize=5,
                    colors=('tab:green', 'tab:blue'),
                    title='Feature Importance Comparison',
                    title_size=20,
                    tick_size=16,
                    tick_rotation=None,
                    label_size=16,
                    legend_loc=None,
                    figsize=(10, 7),
                    return_fig=False):
    '''
    Plot comparison between two different Explanation objects.

    Args:
      comparison_explanations: tuple of Explanation objects to be compared.
      comparison_names: tuple of names for each Explanation object.
      feature_names: list of feature names.
      sort_features: whether to sort features by their SAGE values.
      max_features: number of features to display.
      orientation: horizontal (default) or vertical.
      error_bars: whether to include standard deviation error bars.
      confidence_level: confidence interval coverage (e.g., 95%).
      capsize: error bar cap width.
      colors: colors for each set of SAGE values.
      title: plot title.
      title_size: font size for title.
      tick_size: font size for feature names and numerical values.
      tick_rotation: tick rotation for feature names (vertical plots only).
      label_size: font size for label.
      legend_loc: legend location.
      figsize: figure size (if fig is None).
      return_fig: whether to return matplotlib figure object.
    '''
    # Default feature names.
    if feature_names is None:
        feature_names = ['Feature {}'.format(i) for i in
                         range(len(comparison_explanations[0].values))]

    # Default comparison names.
    num_comps = len(comparison_explanations)
    if num_comps not in (2, 3, 4, 5):
        raise ValueError('only support comparisons for 2-5 explanations')
    if comparison_names is None:
        comparison_names = ['Explanation {}'.format(i) for i in
                            range(num_comps)]

    # Default colors.
    if colors is None:
        colors = ['tab:green', 'tab:blue', 'tab:purple',
                  'tab:orange', 'tab:pink'][:num_comps]

    # Determine explanation type.
    unique_types = np.unique([explanation.explanation_type
                              for explanation in comparison_explanations])
    if len(unique_types) == 1:
        explanation_type = unique_types[0]
    else:
        explanation_type = 'Importance'

    # Sort features if necessary.
    if len(feature_names) > max_features:
        sort_features = True

    # Extract values.
    values = [sage_values.values for sage_values in comparison_explanations]
    std = [sage_values.std for sage_values in comparison_explanations]

    # Perform sorting.
    if sort_features:
        argsort = np.argsort(values[0])[::-1]
        values = [sage_values[argsort] for sage_values in values]
        std = [stddev[argsort] for stddev in std]
        feature_names = np.array(feature_names)[argsort]

    # Remove extra features if necessary.
    if len(feature_names) > max_features:
        feature_names = (list(feature_names[:max_features])
                         + ['Remaining Features'])
        values = [
            list(sage_values[:max_features])
            + [np.sum(sage_values[max_features:])]
            for sage_values in values]
        std = [list(stddev[:max_features])
               + [np.sum(stddev[max_features:] ** 2) ** 0.5]
               for stddev in std]

    # Warn if too many features.
    if len(feature_names) > 50:
        warnings.warn('Plotting {} features may make figure too crowded, '
                      'consider using max_features'.format(
                        len(feature_names)), Warning)

    # Discard std if necessary.
    if not error_bars:
        std = [None for _ in std]
    else:
        assert 0 < confidence_level < 1
        std = [stddev * norm.ppf(0.5 + confidence_level / 2) for stddev in std]

    # Make plot.
    width = 0.8 / num_comps
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()

    if orientation == 'horizontal':
        # Bar chart.
        enumeration = enumerate(zip(values, std, comparison_names, colors))
        for i, (sage_values, stddev, name, color) in enumeration:
            pos = - 0.4 + width / 2 + width * i
            ax.barh(np.arange(len(feature_names))[::-1] - pos,
                    sage_values, height=width, color=color, xerr=stddev,
                    capsize=capsize, label=name)

        # Feature labels.
        if tick_rotation is not None:
            raise ValueError('rotation not supported for horizontal charts')
        ax.set_yticks(np.arange(len(feature_names))[::-1])
        ax.set_yticklabels(feature_names, fontsize=label_size)

        # Axis labels and ticks.
        ax.set_ylabel('')
        ax.set_xlabel('{} value'.format(explanation_type), fontsize=label_size)
        ax.tick_params(axis='x', labelsize=tick_size)

    elif orientation == 'vertical':
        # Bar chart.
        enumeration = enumerate(zip(values, std, comparison_names, colors))
        for i, (sage_values, stddev, name, color) in enumeration:
            pos = - 0.4 + width / 2 + width * i
            ax.bar(np.arange(len(feature_names)) + pos,
                   sage_values, width=width, color=color, yerr=stddev,
                   capsize=capsize, label=name)

        # Feature labels.
        if tick_rotation is None:
            tick_rotation = 45
        if tick_rotation < 90:
            ha = 'right'
            rotation_mode = 'anchor'
        else:
            ha = 'center'
            rotation_mode = 'default'
        ax.set_xticks(np.arange(len(feature_names)))
        ax.set_xticklabels(feature_names, rotation=tick_rotation, ha=ha,
                           rotation_mode=rotation_mode,
                           fontsize=label_size)

        # Axis labels and ticks.
        ax.set_ylabel('{} value'.format(explanation_type), fontsize=label_size)
        ax.set_xlabel('')
        ax.tick_params(axis='y', labelsize=tick_size)

    # Remove spines.
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.legend(loc=legend_loc, fontsize=label_size)
    ax.set_title(title, fontsize=title_size)
    plt.tight_layout()

    if return_fig:
        return fig
    else:
        return
