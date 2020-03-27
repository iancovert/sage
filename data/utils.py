from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import train_test_split


class TabularDataset(Dataset):
    '''
    Dataset wrapper, capable of using subset of inputs and outputs.
    '''
    def __init__(self,
                 data,
                 labels):
        self.input_size = data.shape[1]
        self.data = data.astype(np.float32)
        if len(labels.shape) == 1:
            self.output_size = len(np.unique(labels))
            self.labels = labels.astype(np.long)
        else:
            self.output_size = labels.shape[1]
            self.labels = labels.astype(np.float32)
        self.set_inds(None)
        self.set_output_inds(None)

    def set_inds(self, inds):
        data = self.data
        if inds is not None:
            inds = np.array([i in inds for i in range(self.input_size)])
            data = data[:, inds]
        self.input = data

    def set_output_inds(self, inds):
        output = self.labels
        if inds is not None:
            assert len(output.shape) == 2
            inds = np.array([i in inds for i in range(self.output_size)])
            output = output[:, inds]
        self.output = output

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.input[index], self.output[index]


def split_data(data, seed=123, val_portion=0.1, test_portion=0.1):
    N = data.shape[0]
    N_val = int(val_portion * N)
    N_test = int(test_portion * N)
    train, test = train_test_split(data, test_size=N_test, random_state=seed)
    train, val = train_test_split(train, test_size=N_val, random_state=seed+1)
    return train, val, test

