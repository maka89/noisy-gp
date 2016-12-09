# noisy-gp
Gaussian Process Regression with noisy inputs and outputs. Variance estimates for the training set X and y, is supplied to the fit() method, and can vary from sample to sample.

Using noisy inputs can give the LML many local maxima. Thus it is useful to supply good bounds for the hyperparameters and run the optimization many times with different initial values. Or use another method to choose hyperparameters.

Papers:
Learning Gaussian Process Models from Uncertain Data - Patrick Dallaire, Camille Besse, and Brahim Chaib-draa
Approximate Methods for Propagation of Uncertainty with Gaussian Process Model. PhD thesis, University of Glasgow, Glasgow, UK (2004)
