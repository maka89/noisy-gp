# noisy-gp
Gaussian Process Regression with noisy inputs and outputs. Variance estimates for the training set X and y, is supplied to the fit() method, and can vary from sample to sample.

Using noisy inputs can give the LML local maxima, this it is useful to supply good bounds for the hyperparameters and restart the optimization many times. Or use a global method to choose hyperparameters.
