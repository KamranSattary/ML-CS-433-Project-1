from helpers import *
from implementations import least_squares_GD, least_squares_SGD, least_squares, \
    ridge_regression, logistic_regression, reg_logistic_regression
from proj1_helpers import *

# define constants
DATA_TRAIN_PATH = 'data/train.csv'
GAMMA = 0.001
MAX_ITER = 4


def print_predition(function, params, test):
    w, loss = function(*params)

    test_labels, tX_test = test
    pred_labels = predict_labels(w, tX_test)
    print(pred_labels == test_labels)
    correct = pred_labels == test_labels
    print(np.sum(np.where(correct == True)))


def main():
    y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
    test_labels, tX_test, _ = load_csv_data(DATA_TRAIN_PATH)

    init_weights = np.random.random_sample((tX.shape[1], 1))
    print_predition(least_squares_GD, (y, tX[:len(tX) // 20], init_weights, MAX_ITER, GAMMA), (test_labels, tX_test))


if __name__ == '__main__':
    main()
