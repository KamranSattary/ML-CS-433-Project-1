from helpers import *
from implementations import least_squares_GD, least_squares_SGD, least_squares, \
    ridge_regression, logistic_regression, reg_logistic_regression, lasso_reg
from proj1_helpers import *

# define constants
DATA_TRAIN_PATH = 'data/train.csv'
GAMMA = 0.0001
MAX_ITER = 100
LAMBDA = 200


def print_predition(function, params, test):
    w, loss = function(*params)

    print(w)

    test_labels, tX_test = test
    pred_labels = predict_labels(w, tX_test)
    correct = np.equal(pred_labels, test_labels)
    correct = correct.reshape(len(correct))
    print(correct)
    print(np.sum(correct), len(test_labels))


def main():
    y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
    y = y.reshape(-1, 1)
    test_labels, tX_test, _ = load_csv_data(DATA_TRAIN_PATH)
    test_labels = test_labels.reshape(-1, 1)
    w, loss = ridge_regression(y, tX, LAMBDA)

    print(w)
    init_weights = np.random.random_sample((tX.shape[1], 1))
    w, _ = lasso_reg(y, tX, init_weights, 100, GAMMA, LAMBDA)

    removed_features = np.where(w == 0)
    tX = np.delete(tX, removed_features, axis=1)
    tX_test = np.delete(tX_test, removed_features, axis=1)
    init_weights = np.random.random_sample((tX.shape[1], 1))

    print_predition(ridge_regression, (y, tX, LAMBDA), (test_labels, tX_test))


if __name__ == '__main__':
    main()
