import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.metrics import mean_squared_error

class HistogramBasedDecisionTree:
    """An implementation of a histogram-based decision tree for regression.

    Parameters:
        max_depth (int): The maximum depth of the decision tree (default: 3).
        n_bins (int): The number of bins used for feature discretization (default: 10).

    Attributes:
        max_depth (int): The maximum depth of the decision tree.
        n_bins (int): The number of bins used for feature discretization.
        tree (dict): The decision tree represented as a dictionary.
    """

    def __init__(self, max_depth=3, n_bins=10):
        """Initialize a HistogramBasedDecisionTree object with the specified parameters.

        Args:
            max_depth (int, optional): The maximum depth of the decision tree (default: 3).
            n_bins (int, optional): The number of bins used for feature discretization (default: 10).
        """
        self.max_depth = max_depth
        self.n_bins = n_bins
        self.tree = {}

    def fit(self, X, y, depth=0, node_id=0):
        """Fit the decision tree to the given training data.

        Args:
            X (array-like): The input features of the training data.
            y (array-like): The target values of the training data.
            depth (int, optional): The current depth of the tree (default: 0).
            node_id (int, optional): The ID of the current tree node (default: 0).
        """
        n_samples, n_features = X.shape

        if depth == self.max_depth or n_samples < 2:
            self.tree[node_id] = np.mean(y)
            return

        best_gain = 0
        best_split = None
        best_left_indices = None
        best_right_indices = None

        for feature_index in range(n_features):
            feature_values = X[:, feature_index]
            bins = np.linspace(np.min(feature_values), np.max(feature_values), self.n_bins)
            for threshold in bins:
                left_indices = np.where(feature_values <= threshold)[0]
                right_indices = np.where(feature_values > threshold)[0]

                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue

                left_std = np.std(y[left_indices])
                right_std = np.std(y[right_indices])
                gain = np.std(y) - (len(left_indices) * left_std + len(right_indices) * right_std) / n_samples

                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature_index, threshold)
                    best_left_indices = left_indices
                    best_right_indices = right_indices

        if best_gain == 0:
            self.tree[node_id] = np.mean(y)
            return

        self.tree[node_id] = best_split
        self.fit(X[best_left_indices], y[best_left_indices], depth + 1, node_id * 2 + 1)
        self.fit(X[best_right_indices], y[best_right_indices], depth + 1, node_id * 2 + 2)

    def predict_sample(self, x, node_id=0):
        """Predict the target value for a single input sample.

        Args:
            x (array-like): The input features of the sample.
            node_id (int, optional): The ID of the current tree node (default: 0).

        Returns:
            float: The predicted target value.
        """
        node_value = self.tree.get(node_id)

        if node_value is None:
            return 0

        if isinstance(node_value, tuple):
            feature_index, threshold = node_value
            if x[feature_index] <= threshold:
                return self.predict_sample(x, node_id * 2 + 1)
            else:
                return self.predict_sample(x, node_id * 2 + 2)
        else:
            return node_value

    def predict(self, X):
        """Make predictions for the given input features.

        Args:
            X (array-like): The input features for making predictions.

        Returns:
            array-like: The predicted target values.
        """
        return np.array([self.predict_sample(x) for x in X])
    
    class LightGBMScratch:
    """An implementation of the LightGBM algorithm from scratch.

    Parameters:
        n_estimators (int): The number of boosting iterations (default: 10).
        learning_rate (float): The learning rate or shrinkage factor (default: 0.1).
        max_depth (int): The maximum depth of the histogram-based decision trees (default: 3).
        n_bins (int): The number of bins used for feature discretization (default: 10).

    Attributes:
        n_estimators (int): The number of boosting iterations.
        learning_rate (float): The learning rate or shrinkage factor.
        max_depth (int): The maximum depth of the histogram-based decision trees.
        n_bins (int): The number of bins used for feature discretization.
        trees (list): The list of histogram-based decision trees in the ensemble.
        training_errors (list): The training errors (mean squared error) at each iteration.
        test_errors (list): The test errors (mean squared error) at each iteration.
    """

    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=3, n_bins=10):
        """Initialize a LightGBMScratch object with the specified parameters.

        Args:
            n_estimators (int, optional): The number of boosting iterations (default: 10).
            learning_rate (float, optional): The learning rate or shrinkage factor (default: 0.1).
            max_depth (int, optional): The maximum depth of the histogram-based decision trees (default: 3).
            n_bins (int, optional): The number of bins used for feature discretization (default: 10).
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_bins = n_bins
        self.trees = []
        self.training_errors = []
        self.test_errors = []

    def fit(self, X, y, X_test=None, y_test=None):
        """Fit the LightGBM model to the given training data.

        Args:
            X (array-like): The input features of the training data.
            y (array-like): The target values of the training data.
            X_test (array-like, optional): The input features of the test data (default: None).
            y_test (array-like, optional): The target values of the test data (default: None).
        """
        residuals = y
        for _ in range(self.n_estimators):
            tree = HistogramBasedDecisionTree(max_depth=self.max_depth, n_bins=self.n_bins)
            tree.fit(X, residuals)
            self.trees.append(tree)
            predictions = tree.predict(X)
            residuals = residuals - self.learning_rate * predictions
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
        return np.sum([self.learning_rate * tree.predict(X) for tree in self.trees], axis=0)