import pickle
import numpy as np
from sage import plotting


class Explanation:
    '''For storing and plotting SAGE values.'''
    def __init__(self, values, std, explanation_type='SAGE'):
        self.values = values
        self.std = std
        self.explanation_type = explanation_type

    def plot(self,
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
          feature_names: list of feature names.
          sort_features: whether to sort features by their SAGE values.
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
        return plotting.plot(
            self, feature_names, sort_features, max_features, orientation,
            error_bars, confidence_level, capsize, color, title, title_size,
            tick_size, tick_rotation, label_size, figsize, return_fig)

    def comparison(self,
                   other_values,
                   comparison_names=None,
                   feature_names=None,
                   sort_features=True,
                   max_features=np.inf,
                   orientation='vertical',
                   error_bars=True,
                   confidence_level=0.95,
                   capsize=5,
                   colors=None,
                   title='Feature Importance Comparison',
                   title_size=20,
                   tick_size=16,
                   tick_rotation=None,
                   label_size=16,
                   legend_loc=None,
                   figsize=(10, 7),
                   return_fig=False):
        '''
        Plot comparison with another set of SAGE values.

        Args:
          other_values: another SAGE values object.
          comparison_names: tuple of names for each SAGE value object.
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
        return plotting.comparison_plot(
            (self, other_values), comparison_names, feature_names,
            sort_features, max_features, orientation, error_bars,
            confidence_level, capsize, colors, title, title_size, tick_size,
            tick_rotation, label_size, legend_loc, figsize, return_fig)

    def save(self, filename):
        '''Save Explanation object.'''
        if isinstance(filename, str):
            with open(filename, 'wb') as f:
                pickle.dump(self, f)
        else:
            raise TypeError('filename must be str')

    def __repr__(self):
        with np.printoptions(precision=2, threshold=12, floatmode='fixed'):
            return '{} Explanation(\n  (Mean): {}\n  (Std):  {}\n)'.format(
                self.explanation_type, self.values, self.std)


def load(filename):
    '''Load Explanation object.'''
    with open(filename, 'rb') as f:
        sage_values = pickle.load(f)
        if isinstance(sage_values, Explanation):
            return sage_values
        else:
            raise ValueError('object is not instance of Explanation class')
