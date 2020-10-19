"""some helper functions."""
import numpy as np
import matplotlib.pyplot as plt


# didnt find how to load from zip with just numpy?
def load_data(path_dataset='../data/', standardize = True):

    """
    Loads data
    :param path_dataset: data folder containing test.csv and train.csv
    :return: train_y (n,), train_tx (n,d+1), test_tx (m,d+1)
    """

    master = np.genfromtxt('../data/train.csv', delimiter=",", skip_header=1,
                           converters={1: lambda x: float(0) if b"b" in x else float(1)})
    train_y, train_x = master[:, 1], np.delete(master, [0, 1], axis=1)

    test_x = np.genfromtxt('../data/test.csv', delimiter=",", skip_header=1)
    test_x = np.delete(test, [0, 1], axis=1)

    if standardize == True:
        xmean, xstd = np.mean(train_tx, axis=0), np.std(train_tx, axis=0)
        train_x = (train_x - xmean) / xstd
        test_x = (test_x - xmean) / xstd

    train_tx = np.c_[np.ones(master.shape[0]), train_x]
    test_tx = np.c_[np.ones(test_x.shape[0]), test_x]

    return train_y, train_tx, test_tx

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):

    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """

    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

