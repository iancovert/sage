import torch.nn as nn


def activation_helper(activation):
    '''Get activation function.'''
    if activation == 'sigmoid':
        act = nn.Sigmoid()
    elif activation == 'tanh':
        act = nn.Tanh()
    elif activation == 'relu':
        act = nn.ReLU()
    elif activation == 'softplus':
        act = nn.Softplus()
    elif activation == 'elu':
        act = nn.ELU()
    elif activation == 'softmax':
        act = nn.Softmax(dim=-1)
    elif activation is None:
        act = nn.Identity()
    else:
        raise ValueError('unsupported activation: {}'.format(activation))
    return act


class MLP(nn.Module):
    '''
    Multilayer perceptron (MLP).

    Args:
      input_size: input features.
      output_size: output dimensionality.
      hidden: number of hidden layers.
      activation: nonlinearity between hidden layers.
      output_activation: nonlinearity at output layer.
    '''
    def __init__(self,
                 input_size,
                 output_size,
                 hidden,
                 activation,
                 output_activation=None):
        super(MLP, self).__init__()

        # Fully connected layers.
        self.input_size = input_size
        self.output_size = output_size
        fc_layers = [nn.Linear(d_in, d_out) for d_in, d_out in
                     zip([input_size] + hidden, hidden + [output_size])]
        self.fc = nn.ModuleList(fc_layers)

        # Activation functions.
        self.activation = activation_helper(activation)
        self.output_activation = activation_helper(output_activation)

    def forward(self, x):
        for i, fc in enumerate(self.fc):
            if i > 0:
                x = self.activation(x)
            x = fc(x)

        return self.output_activation(x)

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.input_size, self.output_size)

