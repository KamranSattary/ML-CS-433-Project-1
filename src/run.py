from helpers import *
from implementations import ridge_regression
from proj1_helpers import *
import argparse
import pickle

# define constants
DATA_TRAIN_PATH = '../data/train.csv'
DATA_TEST_PATH = '../data/test.csv'
RESULT_PATH = '../data/infered.csv'

# define best parameters found
DEGREE = 13
LAMBDA = 1e-20
K_FOLD = 7

# for same k-fold
SEED = 12

# For jupyter notebook
TR_ACCURACY = None
TE_ACCURACY = None


def set_lambda(val):
    """Helper function for the notebook - set LAMBDA
    :param val: new lambda value """
    global LAMBDA
    LAMBDA = val


def save_classif_percentage(tr):
    """This function is used to save the percentage of good classification on the test and training dataset to the
    python notebook
    """
    global TR_ACCURACY

    TR_ACCURACY = tr


def ret_classif_percentage():
    """This function is used to rerurn the percentage of good classification on the test and training dataset to the
    python notebook
    """
    return TR_ACCURACY


def get_normalization_methods(tX):
    """
    Return the normalization methods initialized using training set
    :param: tX: matrix(n, d) training set
    :return: a list of methods that will be applied on data sets
    """
    # remove NaN from input data
    fill_nan = fill_nan_closure(tX)

    # prepare minmax normalization
    xmin, xmax = np.min(tX, axis=0), np.max(tX, axis=0)
    minmax = minmax_normalize_closure(xmax, xmin)

    return [fill_nan, minmax]


def normalize_data(tX, methods):
    """
    Apply normalization methods on the dataset
    :param tX: matrix(n, d) dataset
    :param methods: array(k): methods to be applied
    :return: normalized tX
    """
    for method in methods:
        tX = method(tX)
    return tX


def append_to_lists(lists, values):
    for i in range(len(lists)):
        lists[i].append(values[i].tolist())


def average_lists(lists):
    rets = []
    for lis in lists:
        rets.append(np.mean(lis, axis=0))
    return rets


def train(tX, y):
    """
    Train the model
    :param tX: model(n, d) - input
    :param y: array(n) - labels
    :return w: the weight of the model
    """
    # Split the dataset k-fold
    k_indices = build_k_indices(len(y), K_FOLD, SEED)

    ws_tmp = []
    loss_te_tmp = []
    loss_tr_tmp = []

    for k in range(K_FOLD):
        training, testing = cross_validation(y, tX, k_indices, k)

        # build the polynomial fct
        tx_tr = build_poly(training[0], DEGREE)
        tx_te = build_poly(testing[0], DEGREE)

        # Train the model
        w, _ = ridge_regression(training[1], tx_tr, LAMBDA)

        # Compute loss
        pred_tr = predict(tx_tr, w)
        pred_te = predict(tx_te, w)

        loss_tr = np.sum(np.equal(pred_tr, training[1])) / len(pred_tr)
        loss_te = np.sum(np.equal(pred_te, testing[1])) / len(pred_te)

        append_to_lists((ws_tmp, loss_tr_tmp, loss_te_tmp), (w, loss_tr, loss_te))

    w, good_tr, good_te = average_lists((ws_tmp, loss_tr_tmp, loss_te_tmp))

    save_classif_percentage(good_tr)
    print("Training set good classification {}".format(good_tr))

    return w


def save_predict(tX, w, inds):
    """
    Save predictions
    :param tX: matrix(n, d) data to be predicted
    :param w: array(d) model weights
    :param inds;
    "return None
    """
    pred = predict(build_poly(tX, DEGREE), w)

    create_csv_submission(inds, pred, args.result_path)


def main():
    # Get training set and compute normalization params
    y, tX, ids = load_csv_data(args.train_path)
    norm_method = get_normalization_methods(tX)

    if args.w is None:
        # No saved weights => normalize and train
        tX = normalize_data(tX, norm_method)
        w = train(tX, y)
    else:
        # Otherwise load the saved weight
        with open(args.w, 'rb') as f:
            w = pickle.load(f)

    # Get an normalized pred set + run inference and save
    _, tX_pred, ids_pred = load_csv_data(args.infer_path)
    tX_pred = normalize_data(tX_pred, norm_method)
    save_predict(tX_pred, w, ids_pred)


if __name__ == '__main__':
    # command line arguments
    parser = argparse.ArgumentParser(description='Train(optional) model and generate prediction')
    parser.add_argument('-train_path', default=DATA_TRAIN_PATH, help='Train dataset full path; otherwise assumed')
    parser.add_argument('-infer_path', default=DATA_TEST_PATH, help='Predict dataset full path; otherwise assumed')
    parser.add_argument('-result_path', default=RESULT_PATH, help='Where to save the inference result full path; '
                                                                  'otherwise assumed')
    parser.add_argument('-w', default=None,
                        help='Path to pickle file of weights. If present just the predict step will '
                             'take place')

    args = parser.parse_args()

    main()
