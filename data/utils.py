from sklearn.model_selection import train_test_split


def split_data(data, seed=123, val_portion=0.1, test_portion=0.1):
    N = data.shape[0]
    N_val = int(val_portion * N)
    N_test = int(test_portion * N)
    train, test = train_test_split(data, test_size=N_test, random_state=seed)
    train, val = train_test_split(train, test_size=N_val, random_state=seed+1)
    return train, val, test
