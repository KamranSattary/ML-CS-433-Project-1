from helpers import *
from implementations import least_squares_GD, least_squares_SGD, least_squares, \
    ridge_regression, logistic_regression, reg_logistic_regression, lasso_reg
from proj1_helpers import *
from nn import *

from neural_net import NeuralNetwork

# define constants
DATA_TRAIN_PATH = 'data/train.csv'
DATA_TEST_PATH = 'data/test.csv'

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
    print(y.shape, tX.shape)
    test_labels, tX_test, _ = load_csv_data(DATA_TEST_PATH)
    test_labels = test_labels.reshape(-1, 1)
    w, loss = ridge_regression(y, tX, LAMBDA)

    print(w)
    init_weights = np.random.random_sample((tX.shape[1], 1))
    w, _ = lasso_reg(y.reshape((-1, 1)), tX, init_weights, 40, GAMMA, LAMBDA)

    removed_features = np.where(w == 0)
    tX = np.delete(tX, removed_features, axis=1)
    tX_test = np.delete(tX_test, removed_features, axis=1)

    shape = len(tX[0])
    print(tX.shape, y.shape)

    NN_ARCHITECTURE = [
        {"input_dim": shape, "output_dim": 25, "activation": "relu"},
        {"input_dim": 25, "output_dim": 50, "activation": "relu"},
        {"input_dim": 50, "output_dim": 50, "activation": "relu"},
        {"input_dim": 50, "output_dim": 25, "activation": "relu"},
        {"input_dim": 25, "output_dim": 1, "activation": "sigmoid"},
    ]

    params_values, cost_history, accuracy_history = train(tX[:10000].T, y[:10000].reshape((-1, 1)).T, NN_ARCHITECTURE, 1000, 0.01)
    Y_test_hat, _ = full_forward_propagation(np.transpose(tX_test.T), params_values, NN_ARCHITECTURE)

    print(Y_test_hat)
    acc_test = get_accuracy_value(Y_test_hat, np.transpose(test_labels))
    print("Test set accuracy: {:.2f} - David".format(acc_test))

    # print_predition(ridge_regression, (y, tX, LAMBDA), (test_labels, tX_test))


if __name__ == '__main__':
    main()
