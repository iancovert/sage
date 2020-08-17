import pickle
import numpy as np
import matplotlib.pyplot as plt


class SAGE:
    '''For storing and plotting SAGE values.'''
    def __init__(self, values, std):
        self.values = values
        self.std = std

    def plot(self,
             feature_names,
             sort_features=True,
             max_features=np.inf,
             figsize=(10, 7),
             orientation='horizontal',
             error_bars=True,
             color='tab:green',
             title='Feature Importance',
             title_size=20,
             tick_size=16,
             label_size=16,
             label_rotation=None,
             return_fig=False):
        '''
        Plot SAGE values.

        Args:
          feature_names: list of feature names.
          sort_features: whether to sort features by their SAGE values.
          max_features: number of features to display.
          fig: matplotlib figure (optional).
          figsize: figure size (if fig is None).
          orientation: horizontal (default) or vertical.
          error_bars: whether to include standard deviation error bars.
          color: bar chart color.
          title: plot title.
          title_size: font size for title.
          tick_size: font size for feature names and numerical values.
          label_size: font size for label.
          label_rotation: label rotation (for vertical plots only).
          return_fig: whether to return matplotlib figure object.
        '''
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()

        # Sort features if necessary.
        if len(feature_names) > max_features:
            sort_features = True
        values = self.values
        std = self.std
        if sort_features:
            argsort = np.argsort(values)[::-1]
            values = values[argsort]
            std = std[argsort]
            feature_names = np.array(feature_names)[argsort]

        # Remove extra features if necessary.
        if len(feature_names) > max_features:
            feature_names = list(feature_names[:max_features]) + ['Remaining']
            values = (list(values[:max_features])
                      + [np.sum(values[max_features:])])
            std = (list(std[:max_features])
                   + [np.sum(std[max_features:] ** 2) ** 0.5])

        # Make plot.
        if orientation == 'horizontal':
            # Bar chart.
            if error_bars:
                ax.barh(np.arange(len(feature_names))[::-1], values,
                        color=color, xerr=std)
            else:
                ax.barh(np.arange(len(feature_names))[::-1], values,
                        color=color)

            # Feature labels.
            if label_rotation is not None:
                raise ValueError('rotation not supported for horizontal charts')
            ax.set_yticks(np.arange(len(feature_names))[::-1])
            ax.set_yticklabels(feature_names, fontsize=label_size)

            # Axis labels and ticks.
            ax.set_ylabel('')
            ax.set_xlabel('SAGE value', fontsize=label_size)
            ax.tick_params(axis='x', labelsize=tick_size)

        elif orientation == 'vertical':
            # Bar chart.
            if error_bars:
                ax.bar(np.arange(len(feature_names)), values, color=color,
                       yerr=std)
            else:
                ax.bar(np.arange(len(feature_names)), values, color=color)

            # Feature labels.
            if label_rotation is None:
                label_rotation = 90
            if label_rotation < 90:
                ha = 'right'
                rotation_mode = 'anchor'
            else:
                ha = 'center'
                rotation_mode = 'default'
            ax.set_xticks(np.arange(len(feature_names)))
            ax.set_xticklabels(feature_names, rotation=label_rotation, ha=ha,
                               rotation_mode=rotation_mode,
                               fontsize=label_size)

            # Axis labels and ticks.
            ax.set_ylabel('SAGE value', fontsize=label_size)
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
