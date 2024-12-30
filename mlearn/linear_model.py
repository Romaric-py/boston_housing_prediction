#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 15:03:58 2024

@author: romaric
"""

import numpy as np
import matplotlib.pyplot as plt
import json

from dataclasses import dataclass


@dataclass
class LinearRegressionSGD:
    """Linear Regression model using gradient descent."""
    learning_rate: float = 0.01
    precision: float = 1e-5
    n_iter: int = 1000
    n_iter_no_change: int = 10
    fit_intercept: bool = True
    early_stopping: bool = True

    def __post_init__(self):
        self._is_fitted = False
        self._n_no_change = 0
        self.coef_ = None
        self.intercept_ = None
        self.history = {'loss': []}
        if not self.early_stopping:
            self.n_iter_no_change = self.n_iter

    def _init_params(self, X):
        """Initialize model parameters."""
        n = X.shape[1]
        W = np.random.randn(n, 1)
        b = np.random.randn() if self.fit_intercept else 0.0
        return W, b

    def _output(self, X, W, b):
        """Calculate the model's output."""
        return X.dot(W) + b

    def _error(self, y_pred, y_true):
        """Calculate the error (difference) between predictions and true values."""
        return y_pred - y_true

    def _loss(self, error):
        """Calculate the Mean Squared Error (MSE)."""
        return 0.5 * np.mean(error**2)

    def _compute_gradient(self, X, error):
        """Calculate the gradient of the loss function."""
        m = X.shape[0]
        gradW = X.T.dot(error) / m
        partial_b = error.mean() if self.fit_intercept else 0.0
        return gradW, partial_b

    def _check_data_shape(self, X, y):
        """Check the shapes of the training data."""
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if y.shape[1] != 1:
            raise ValueError("y must be a vector or matrix of shape (m, 1).")
        if X.shape[0] != y.shape[0]:
            raise ValueError("The number of rows in X and y must be the same.")
        return X, y

    def _check_data_shape_for_prediction(self, X):
        """Check the shape of the data for prediction."""
        if not self.is_fitted():
            raise RuntimeError("The model must be fitted before making predictions.")
        if X.ndim != 2 or X.shape[1] != self.coef_.shape[0]:
            raise ValueError("The shape of X does not match the training data.")

    def _stop_training(self):
        """Determine whether the training should stop."""
        if len(self.history['loss']) < 2:
            return False
        last_loss = self.history['loss'][-1]
        prev_loss = self.history['loss'][-2]
        if abs(last_loss - prev_loss) < self.precision:
            self._n_no_change += 1
        else:
            self._n_no_change = 0
        return self._n_no_change >= self.n_iter_no_change

    def fit(self, X, y, verbose=False):
        """Train the model."""
        X, y = self._check_data_shape(X, y)
        W, b = self._init_params(X)
        self.history['loss'] = []

        for i in range(self.n_iter):
            y_pred = self._output(X, W, b)
            error = self._error(y_pred, y)
            loss = self._loss(error)
            self.history['loss'].append(loss)

            gradW, partial_b = self._compute_gradient(X, error)
            W -= self.learning_rate * gradW
            b -= self.learning_rate * partial_b

            if verbose:
                print(f"Iteration {i + 1}, Loss: {loss:.6f}")

            if self.early_stopping and self._stop_training():
                if verbose:
                    print(f"Early stopping after {i + 1} iterations.")
                break

        self.coef_ = W
        self.intercept_ = b
        self._is_fitted = True

    def is_fitted(self):
        return self._is_fitted

    def predict(self, X):
        """Predict the values for X."""
        self._check_data_shape_for_prediction(X)
        return self._output(X, self.coef_, self.intercept_).flatten()

    def plot(self):
        """Plot the loss curve."""
        if not self.history['loss']:
            raise ValueError("No loss data available for plotting.")
        plt.plot(self.history['loss'])
        plt.title("Loss Curve")
        plt.xlabel("Iterations")
        plt.ylabel("Loss (MSE)")
        plt.show()

    def to_json(self, path):
        """Save the model to a JSON file."""
        if not self.is_fitted():
            raise RuntimeError("The model must be fitted before saving.")
        with open(path, 'w') as f:
            json.dump({
                'coef_': self.coef_.tolist(),
                'intercept_': self.intercept_.tolist()
            }, f, indent=4)

    def load_from_json(self, path):
        """Load a model from a JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        self.coef_ = np.array(data['coef_'])
        self.intercept_ = np.array(data['intercept_'])
        self._is_fitted = True

#####################################################################

class LogisticRegressionSGD:
    """Logistic Regression classifier using gradient descent."""

    def __init__(self, learning_rate=0.01, precision=1e-5, n_iter=1000, n_iter_no_change=10, fit_intercept=True, early_stopping=True):
        self.learning_rate = learning_rate
        self.precision = precision
        self.n_iter = n_iter
        self.n_iter_no_change = n_iter_no_change
        self.fit_intercept = fit_intercept
        self.early_stopping = early_stopping
        self._is_fitted = False
        self.coef_ = None
        self.intercept_ = None
        self.history = {'loss': []}
        self._n_no_change = 0

    def _sigmoid(self, z):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-z))

    def _init_params(self, X):
        """Initialize the model parameters."""
        n = X.shape[1]
        coef_ = np.random.randn(n, 1)
        intercept_ = np.random.randn(1)
        return coef_, intercept_

    def _output(self, X, coef_, intercept_):
        """Compute the linear model's output."""
        return self._sigmoid(X.dot(coef_) + intercept_)

    def _error(self, y_pred, y_true):
        """Calculate the error (difference) between predictions and true values."""
        return y_pred - y_true

    def _loss(self, error, y_true):
        """Compute the logistic loss function (cross-entropy loss)."""
        m = len(y_true)
        loss = - (1 / m) * np.sum(y_true * np.log(error) + (1 - y_true) * np.log(1 - error))
        return loss

    def _compute_gradient(self, X, error, y_true):
        """Compute the gradients of the loss function."""
        m = X.shape[0]
        grad_coef = (1 / m) * X.T.dot(error)
        grad_intercept = np.mean(error)
        return grad_coef, grad_intercept

    def _check_data_shape(self, X, y):
        """Check the shape of the training data."""
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        return X, y

    def _check_data_shape_for_prediction(self, X):
        """Check the shape of the input data for prediction."""
        if not self.is_fitted():
            raise RuntimeError("The model must be fitted before making predictions.")
        if X.ndim != 2 or X.shape[1] != self.coef_.shape[0]:
            raise ValueError("The shape of X does not match the training data.")

    def _stop_training(self):
        """Determine whether training should stop (early stopping)."""
        if len(self.history['loss']) < 2:
            return False
        last_loss = self.history['loss'][-1]
        prev_loss = self.history['loss'][-2]
        if abs(last_loss - prev_loss) < self.precision:
            self._n_no_change += 1
        else:
            self._n_no_change = 0
        return self._n_no_change >= self.n_iter_no_change

    def fit(self, X, y, verbose=False):
        """Train the logistic regression model using gradient descent."""
        X, y = self._check_data_shape(X, y)
        coef_, intercept_ = self._init_params(X)
        self.history['loss'] = []  # Reset history

        for i in range(self.n_iter):
            y_pred = self._output(X, coef_, intercept_)
            error = self._error(y_pred, y)
            loss = self._loss(y_pred, y)
            self.history['loss'].append(loss)

            grad_coef, grad_intercept = self._compute_gradient(X, error, y)
            coef_ -= self.learning_rate * grad_coef
            intercept_ -= self.learning_rate * grad_intercept

            if verbose:
                print(f"Iteration {i + 1}, Loss: {loss:.6f}")

            if self.early_stopping and self._stop_training():
                if verbose:
                    print(f"Early stopping after {i + 1} iterations.")
                break

        self.coef_ = coef_
        self.intercept_ = intercept_
        self._is_fitted = True

    def is_fitted(self):
        return self._is_fitted

    def predict(self, X):
        """Predict class labels for input data."""
        self._check_data_shape_for_prediction(X)
        y_pred = self._output(X, self.coef_, self.intercept_)
        return (y_pred >= 0.5).astype(int).flatten()

    def plot(self):
        """Plot the loss curve."""
        if not self.history['loss']:
            raise ValueError("No loss data available for plotting.")
        plt.plot(self.history['loss'])
        plt.title("Loss Curve")
        plt.xlabel("Iterations")
        plt.ylabel("Loss (Cross-Entropy)")
        plt.show()

    def to_json(self, path):
        """Save the model to a JSON file."""
        if not self.is_fitted():
            raise RuntimeError("The model must be fitted before saving.")
        with open(path, 'w') as f:
            json.dump({
                'coef_': self.coef_.tolist(),
                'intercept_': self.intercept_.tolist()
            }, f, indent=4)

    def load_from_json(self, path):
        """Load a model from a JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        self.coef_ = np.array(data['coef_'])
        self.intercept_ = np.array(data['intercept_'])
        self._is_fitted = True


###########################################################

LinearRegression = LinearRegressionSGD
LogisticRegression = LogisticRegressionSGD