import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.metrics import mean_squared_error

class XGBoostRegressionTree:
    """A regression tree used in the XGBoost algorithm.

    Attributes:
        feature_index (int): The index of the feature used for splitting.
        threshold (float): The threshold value for splitting.
        left_value (float): The value assigned to the left side of the split.
        right_value (float): The value assigned to the right side of the split.
    """

    def __init__(self):
        """Initialize an XGBoostRegressionTree object with default attribute values."""
        self.feature_index = 0
        self.threshold = 0
        self.left_value = 0
        self.right_value = 0

class XGBoostScratch:
    """Implementation of the XGBoost algorithm for regression.

    Parameters:
        n_estimators (int): The number of boosting iterations (default: 100).
        learning_rate (float): The learning rate or shrinkage factor (default: 0.1).
        reg_lambda (float): The L2 regularization term for leaf weights (default: 1).
        reg_alpha (float): The L1 regularization term for leaf weights (default: 1).

    Attributes:
        n_estimators (int): The number of boosting iterations.
        learning_rate (float): The learning rate or shrinkage factor.
        reg_lambda (float): The L2 regularization term for leaf weights.
        reg_alpha (float): The L1 regularization term for leaf weights.
        trees (list): The list of regression trees in the ensemble.
        training_errors (list): The training errors (mean squared error) at each iteration.
        test_errors (list): The test errors (mean squared error) at each iteration.
        init_prediction (float): The initial prediction made by the ensemble.
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, reg_lambda=1, reg_alpha=1):
        """Initialize an XGBoost object with the specified parameters.

        Args:
            n_estimators (int, optional): The number of boosting iterations (default: 100).
            learning_rate (float, optional): The learning rate or shrinkage factor (default: 0.1).
            reg_lambda (float, optional): The L2 regularization term for leaf weights (default: 1).
            reg_alpha (float, optional): The L1 regularization term for leaf weights (default: 1).
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.trees = []
        self.training_errors = []
        self.test_errors = []
        self.init_prediction = 0

    def fit(self, X, y, X_test=None, y_test=None):
        """Fit the XGBoost model to the given training data.

        Args:
            X (array-like): The input features of the training data.
            y (array-like): The target values of the training data.
            X_test (array-like, optional): The input features of the test data (default: None).
            y_test (array-like, optional): The target values of the test data (default: None).
        """
        self.init_prediction = np.mean(y)
        pred = np.full_like(y, self.init_prediction, dtype=np.float64)
        residuals = y - pred

        for _ in range(self.n_estimators):
            tree = self.fit_stump(X, residuals)
            self.trees.append(tree)

            pred += self.learning_rate * self.predict_stump(X, tree).astype(np.float64)
            residuals = y - pred
            self.training_errors.append(np.mean(residuals ** 2))

            if X_test is not None and y_test is not None:
                test_pred = self.predict(X_test)
                self.test_errors.append(mean_squared_error(y_test, test_pred))

    def predict(self, X):
        """Make predictions for the given input features.

        Args:
            X (array-like): The input features for making predictions.

        Returns:
            array-like: The predicted target values.
        """
        pred = np.full(X.shape[0], self.init_prediction)
        for tree in self.trees:
            pred += self.learning_rate * self.predict_stump(X, tree)
        return pred

    def fit_stump(self, X, y):
        """Fit a regression tree stump to the given data.

        Args:
            X (array-like): The input features of the data.
            y (array-like): The target values of the data.

        Returns:
            XGBoostRegressionTree: The fitted regression tree stump.
        """
        n_samples, n_features = X.shape
        best_feature = None
        best_threshold = None
        best_left_value = None
        best_right_value = None
        best_error = np.inf

        for feature_index in range(n_features):
            feature_values = X[:, feature_index]
            for threshold in feature_values:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask

                left_value = np.sum(y[left_mask]) / (np.sum(left_mask) + self.reg_lambda)
                right_value = np.sum(y[right_mask]) / (np.sum(right_mask) + self.reg_lambda)

                residuals = np.where(left_mask, left_value, right_value)
                error = (
                    np.sum((y - residuals) ** 2)
                    + self.reg_lambda * (left_value ** 2 + right_value ** 2)
                    + self.reg_alpha * (np.abs(left_value) + np.abs(right_value))
                )

                if error < best_error:
                    best_feature = feature_index
                    best_threshold = threshold
                    best_left_value = left_value
                    best_right_value = right_value
                    best_error = error

        stump = XGBoostRegressionTree()
        stump.feature_index = best_feature
        stump.threshold = best_threshold
        stump.left_value = best_left_value
        stump.right_value = best_right_value
        return stump

    def predict_stump(self, X, tree):
        """Make predictions using a regression tree stump.

        Args:
            X (array-like): The input features for making predictions.
            tree (XGBoostRegressionTree): The regression tree stump.

        Returns:
            array-like: The predicted target values.
        """
        return np.where(X[:, tree.feature_index] <= tree.threshold, tree.left_value, tree.right_value)
