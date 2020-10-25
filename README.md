# TEAM KLR - Project 1 - 2020 Fall Machine Learning (CS-433)

[Full assignment guideline pdf here](https://github.com/epfml/ML_course/blob/master/projects/project1/project1_description.pdf)


## Organisation

There are three folders : `data`, `src` and `plot`

### Data

In the `data` folder you will find `train.csv`, `test.csv` as well as `sample-submission.csv`.

- `train.csv` is the data used to train our model.
- `test.csv` is the data that we want to classify thanks to our previously trained model.
- `sample-submission.csv` is the predictions on the `test.csv` dataset.

### Plots

In the `plots` folder you will find some important graphs.

- `Comparing MSE and classification loss.pdf`
- `Determining the best degree.pdf`
- `Distant regressand values.pdf`
- `Zoom determining the best degree.pdf`

### Src

In the `src` folder you will find our code. 

- `run.py` is the code to run in order to get the same `sample-submission.csv` as our best submittion on AIcrowd.com.
- `implementations.py` contains multiple functions used in our model. You will find (among other functions that we used for training our model) the `least_squares_GD(y, tx, initial_w, max_iters, gamma)`, `least_squares_SGD(y, tx, initial_w, max_iters, gamma)`, `least_squares(y, tx)`, `ridge_regression(y, tx, lambda_)`, `logistic_regression(y, tx, initial_w, max_iters, gamma)`, `reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)` required.
- `helpers.py` contains our own useful functions used to handle the data. The provided `proj1_helpers.py` has been left untouched.
- `Visualising Ridge Tuning.ipynb` is the notebook that allowed us to create most of the plots in the folder `plots`, with `select_best_lambda.ipynb` producing the lambda plot. You will find explanations concerning our findings and our thought process in it.

## Data preparation

After importing the training data, an important step is to clean it and to normalize it. A quick look to the data allowed us to see that they were a lot of unnormal -999 values. Therefore, we filled them with NA values and then replaced them with the mediam of their corresponding column.

This being done, we finally normalize the data thanks to the `minmax_normalize_closure` function in `helpers.py`. 

## Feature generation

For feature generation, we used the `built_poly` function in `helpers.py`. This function allows us to realize a feature expansion up to a certain degree for all features in the dataset. 

## Cross validation

The cross validation steps go as follows:

- We retrieve k_indices of test and the rest as training
- We split the data according to determined indices
- We apply the feature expansion on the data with `build_poly`
- We compute optimal weights using `ridge_regression`
- We calculate fold MSE for training and test partitions thanks to `compute_mse`
- We calculate fold misclassification % (which is the ratio of how many inaccurate predictions were made)

## Usage

Run in the terminal:

```
python run.py
```

Then the result should be in the `data` file in `sample-submission.csv`. 

## Report

Here is our report in pdf format: REPORT.PDF

## Team

- Kamran Nejad-Sattary (kamran.nejad-sattary@epfl.ch)
- Razvan Mocan (razvan-florin.mocan@epfl.ch)
- Loïc Busson (loic.busson@epfl.ch)
