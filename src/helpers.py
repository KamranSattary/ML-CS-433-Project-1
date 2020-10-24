"""some helper functions."""
import numpy as np
from implementations import compute_loss, compute_gradient_mse, compute_mse
from proj1_helpers import load_csv_data


def build_poly(x, degree):

    """
    Builds polynomial augmented dataset
    :param x: 
    :param degree: 
    :return:
    """
    r = x.copy()
    for deg in range (2,degree+1):
        r = np.c_[r, np.power(x, deg)]
        
    return np.c_[np.ones(r.shape[0]), r]



def build_k_indices(y, k_fold, seed):

    """
    Builds k index sets from input data
    :param y:
    :param k_fold:
    :param seed:
    :return:
    """

    # set seed for reproducibility and jumble data
    np.random.seed(seed)
    num_row = y.shape[0]
    indices = np.random.permutation(num_row)

    # determine interval length
    interval = int(num_row/k_fold)

    # create index list between start and end of interval of each fold
    k_indices = [indices[k*interval: (k+1)*interval] for k in range(k_fold)]

    return np.array(k_indices)


def fill_nan_closure(x_fill, fill_method=np.nanmedian):
    """
    Replaces missing values given a method (mean, median) to use to calculate replacement
    :param x_fill: matrix(n, d) matrix which NaNs should be filled
    :param fill_method: string of function to use to find fill values, default np.nanmedian, supported are np.nanmedian
    and np.nanmean
    :return x: normalized with given method and nan-filled with given values, closure that knows the fill value for the
    prediction set
    """

    fill_val = fill_method(x_fill, axis=0)

    def fill_nan(x):
        # Retrieve fill values, remember -999 is NaN
        inds = np.where(x == -999)
        x[inds] = np.nan

        # Place column means in the indices. Align the arrays using take
        x[inds] = np.take(fill_val, inds[1])

        return x_fill

    x_fill = fill_nan(x_fill)

    return x_fill, fill_nan


def minmax_normalize_closure(xmax, xmin):
    """
    Normalizes matrix given max x and min x
    :param xmax:
    :param xmin:
    :return: function that only takes x as argument
    """
    def minmax_normalize(x):
        return (x - xmin) / (xmax - xmin)

    return minmax_normalize


def standardize_closure(mu, std):
    """
    Standardizes matrix given mean mu and standard deviation
    :param mu:
    :param std:
    :return:
    """
    def standardize(x):
        return (x - mu) / std

    return standardize

def predict_without_classifying(weights, data):
    #"""Generates predictions (pre-classification via threshold!) given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    
    return y_pred

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    :param y: (n,) array - labels
    :param tx: (n,d) matrix - inputs
    :param batch_size: int
    :param num_batches: int; number of batches
    :param shuffle: boolean; shuffle the dataset or not to avoid ordering in the original data messing with the
    randomness of the minibatches
    :return: final weights vector and loss
    :return: an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """

    data_size = len(y)

    # If !shuffle; easier for processor as branching takes a lot of time
    shuffled_y = y
    shuffled_tx = tx

    if shuffle:
        # If permutation param is an integer, randomly permute ``np.arange(param)``
        shuffle_indices = np.random.permutation(data_size)
        shuffled_y = shuffled_y[shuffle_indices]
        shuffled_tx = shuffled_tx[shuffle_indices]

    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)

        if start_index < end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


# do we use this? if not delete
def lasso_reg(y, tx, initial_w, max_iters, gamma, LAMBDA):
    """
    L1 regularized linear regression = Lasso Regression
    :param y: (n,) array
    :param tx: (n,d) matrix
    :param intial_w: (d,) array; initial weights
    :param max_iters: int; number of iterations
    :param gamma: float; learning rate
    :return: final weights vector and loss
    """

    w = initial_w
    for n_iter in range(max_iters):
        # retrieve gradient and cost
        grd, e = compute_gradient_mse(y, tx, w)

        # prepare the regularization factor
        reg = np.sign(w) * (-1)

        # update the weights
        w = w - (grd + reg * LAMBDA) * gamma

        # set small weights to 0
        w[w < 0.15] = 0
        print(f"Step loss: {compute_mse(e)}")

    # calculate the final loss
    loss = compute_loss(y, tx, w, compute_mse) + np.sum(np.abs(w) * LAMBDA)

    return w, loss
