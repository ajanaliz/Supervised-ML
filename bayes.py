# https://deeplearningcourses.com/c/data-science-supervised-machine-learning-in-python
# https://www.udemy.com/data-science-supervised-machine-learning-in-python
# This is an example of a Bayes classifier on MNIST data.

import numpy as np
from util import get_data
from datetime import datetime
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn

class Bayes(object):
    def fit(self, X, Y, smoothing=10e-3):
        N, D = X.shape
        self.gaussians = dict() # dict of gaussian parameters
        self.priors = dict()
        labels = set(Y) # to get all the unique values in y.
        for c in labels: # loop through each of the labels/classes
            current_x = X[Y == c] # set the current X to the X for which Y is equal to the current class.
            self.gaussians[c] = { # set the mean and the covariance for this gaussian.
                'mean': current_x.mean(axis=0),
				"""for the bayes classifier, we need to calculate and store the covariance instead f the variance.
				notice that, numpy already has a function that will calculate this for us.--> we need to transpose X
				first though, so that it comes out as a D by D matrix, otherwise, numpy will do it the wrong way and
				the result will be N by N. one note, that numpy uses the un-biased version of the covariance, so it 
				divides by N-1 rather than N."""
                'cov': np.cov(current_x.T) + np.eye(D)*smoothing,# add smoothing parameter.
            }
			# calculate the prior.
            self.priors[c] = float(len(Y[Y == c])) / len(Y)

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)

    def predict(self, X):
        N, D = X.shape
        K = len(self.gaussians) # number of classes
        P = np.zeros((N, K)) # for each of the N samples, there are going to be k different probabilities.
        for c, g in self.gaussians.iteritems():
            mean, cov = g['mean'], g['cov']
            P[:,c] = mvn.logpdf(X, mean=mean, cov=cov) + np.log(self.priors[c])
        return np.argmax(P, axis=1) # returns an N sized array.


if __name__ == '__main__':
    X, Y = get_data(10000)
    Ntrain = len(Y) / 2
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

    model = Bayes()
    t0 = datetime.now()
    model.fit(Xtrain, Ytrain)
    print "Training time:", (datetime.now() - t0)

    t0 = datetime.now()
    print "Train accuracy:", model.score(Xtrain, Ytrain)
    print "Time to compute train accuracy:", (datetime.now() - t0), "Train size:", len(Ytrain)

    t0 = datetime.now()
    print "Test accuracy:", model.score(Xtest, Ytest)
    print "Time to compute test accuracy:", (datetime.now() - t0), "Test size:", len(Ytest)
