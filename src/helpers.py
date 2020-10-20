"""some helper functions."""
import numpy as np
import matplotlib.pyplot as plt

# didnt find how to load from zip with just numpy?
def load_data(path_dataset='../data/', standardize = True):

    """
    Loads data
    :param path_dataset: data folder containing test.csv and train.csv
    :param standardize: True or False indicating whether to standardize the data
    :return: train_y (n,), train_x (n,d), test_x (m,d)
    """

    master = np.genfromtxt(path_dataset + 'train.csv', delimiter=",", skip_header=1,
                           converters={1: lambda x: float(0) if b"b" in x else float(1)})
    train_y, train_x = master[:, 1], np.delete(master, [0, 1], axis=1)

    test_x = np.genfromtxt(path_dataset + 'test.csv', delimiter=",", skip_header=1)
    test_x = np.delete(test_x, [0, 1], axis=1)

    if standardize == True:
        xmean, xstd = np.mean(train_x, axis=0), np.std(train_x, axis=0)
        train_x = (train_x - xmean) / xstd
        test_x = (test_x - xmean) / xstd

    return train_y, train_x, test_x


def build_poly(x, degree):
    """
    Builds polynomial augmented data
    :param x:
    :param degree:
    :return: tx
    """
    r = x.copy()
    for deg in range(2, degree + 1):
        r = np.c_[r, np.power(x, deg)]

    tx = np.c_[np.ones(r.shape[0]), r]
    return tx

def build_k_indices(y, k_fold, seed):

    """
    Builds k index sets from input data
    :param y:
    :param k_fold:
    :param seed:
    :return:
    """

    np.random.seed(seed)

    num_row = y.shape[0]
    interval = int(num_row/k_fold)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k*interval: (k+1) * interval] for k in range(k_fold)]

    return np.array(k_indices)

# example of how to use build_k_indices to cross validate
def cross_validation(y, x, k_indices, k, lambda_, degree):
    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    y_te, y_tr = y[te_indice], y[tr_indice]
    x_te, x_tr = x[te_indice], x[tr_indice]

    tx_tr = build_poly(x_tr, degree)
    tx_te = build_poly(x_te, degree)

    w = ridge_regression(y_tr, tx_tr, lambda_)

    loss_tr = np.sqrt(2 * compute_mse(y_tr, tx_tr, w))
    loss_te = np.sqrt(2 * compute_mse(y_te, tx_te, w))

    return loss_tr, loss_te, w

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

