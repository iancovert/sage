import pickle
import numpy as np
from sage import plotting


class SAGE:
    '''For storing and plotting SAGE values.'''
    def __init__(self, values, std):
        self.values = values
        self.std = std

    def plot(self,
             feature_names=None,
             sort_features=True,
             max_features=np.inf,
             orientation='horizontal',
             error_bars=True,
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
            error_bars, color, title, title_size, tick_size, tick_rotation,
            label_size, figsize, return_fig)

    def comparison(self,
                   other_values,
                   comparison_names=None,
                   feature_names=None,
                   sort_features=True,
                   max_features=np.inf,
                   orientation='vertical',
                   error_bars=True,
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
            sort_features, max_features, orientation, error_bars, colors, title,
            title_size, tick_size, tick_rotation, label_size, legend_loc,
            figsize, return_fig)

    def save(self, filename):
        '''Save SAGE object.'''
        if isinstance(filename, str):
            with open(filename, 'wb') as f:
                pickle.dump(self, f)
        else:
            raise TypeError('filename must be str')


def load(filename):
    '''Load SAGE object.'''
    with open(filename, 'rb') as f:
        sage_values = pickle.load(f)
        if isinstance(sage_values, SAGE):
            return sage_values
        else:
            raise ValueError('object is not instance of SAGE class')
