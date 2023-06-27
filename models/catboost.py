import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.metrics import mean_squared_error

class ObliviousTree:
    """An oblivious decision tree model.

    Attributes:
        feature_indices (list): The list of feature indices used for splitting.
        thresholds (list): The list of thresholds for splitting.
        values (list): The list of values assigned to each leaf node.
    """

    def __init__(self):
        """Initialize an ObliviousTree object with empty attribute lists."""
        self.feature_indices = []
        self.thresholds = []
        self.values = []
        
class CatBoostScratch:
    """Implementation of the CatBoost algorithm for regression.

    Parameters:
        n_estimators (int): The number of boosting iterations (default: 100).
        learning_rate (float): The learning rate or shrinkage factor (default: 0.1).
        depth (int): The depth of the oblivious trees (default: 3).

    Attributes:
        n_estimators (int): The number of boosting iterations.
        learning_rate (float): The learning rate or shrinkage factor.
        depth (int): The depth of the oblivious trees.
        trees (list): The list of oblivious trees in the ensemble.
        training_errors (list): The training errors (mean squared error) at each iteration.
        test_errors (list): The test errors (mean squared error) at each iteration.
    """

    def __init__(self, n_estimators=100, learning_rate=0.1, depth=3):
        """Initialize a CatBoost object with the specified parameters.

        Args:
            n_estimators (int, optional): The number of boosting iterations (default: 100).
            learning_rate (float, optional): The learning rate or shrinkage factor (default: 0.1).
            depth (int, optional): The depth of the oblivious trees (default: 3).
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.depth = depth
        self.trees = []
        self.training_errors = []
        self.test_errors = []

    def fit(self, X, y, X_test=None, y_test=None):
        """Fit the CatBoost model to the given training data.

        Args:
            X (array-like): The input features of the training data.
            y (array-like): The target values of the training data.
            X_test (array-like, optional): The input features of the test data (default: None).
            y_test (array-like, optional): The target values of the test data (default: None).
        """
        pred = np.zeros(y.shape)
        residuals = y - pred

        for _ in range(self.n_estimators):
            tree = self.fit_oblivious_tree(X, residuals)
            self.trees.append(tree)

            pred += self.learning_rate * self.predict_oblivious_tree(X, tree)
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
        pred = np.zeros(X.shape[0])
        for tree in self.trees:
            pred += self.learning_rate * self.predict_oblivious_tree(X, tree)
        return pred

    def fit_oblivious_tree(self, X, y):
        """Fit an oblivious tree to the given data.

        Args:
            X (array-like): The input features of the data.
            y (array-like): The target values of the data.

        Returns:
            ObliviousTree: The fitted oblivious tree.
        """
        n_samples, n_features = X.shape
        tree = ObliviousTree()

        for _ in range(self.depth):
            best_feature = None
            best_threshold = None
            best_values = None
            best_error = np.inf

            for feature_index in range(n_features):
                feature_values = X[:, feature_index]
                for threshold in feature_values:
                    paths = np.zeros(X.shape[0], dtype=int)
                    for f_index, t in zip(tree.feature_indices, tree.thresholds):
                        paths = (paths << 1) | (X[:, f_index] > t)
                    new_paths = (paths << 1) | (feature_values > threshold)

                    values = [
                        np.mean(y[new_paths == i]) if np.any(new_paths == i) else 0 for i in range(2 ** (len(tree.feature_indices) + 1))
                    ]

                    residuals = np.array([values[path] for path in new_paths])
                    error = np.mean((y - residuals) ** 2)

                    if error < best_error:
                        best_feature = feature_index
                        best_threshold = threshold
                        best_values = values
                        best_error = error

            tree.feature_indices.append(best_feature)
            tree.thresholds.append(best_threshold)
            tree.values = best_values

        return tree

    def predict_oblivious_tree(self, X, tree):
        """Make predictions using an oblivious tree.

        Args:
            X (array-like): The input features for making predictions.
            tree (ObliviousTree): The oblivious tree.

        Returns:
            array-like: The predicted target values.
        """
        # Create a binary string representation of the path taken for each sample
        paths = np.zeros(X.shape[0], dtype=int)

        for feature_index, threshold in zip(tree.feature_indices, tree.thresholds):
            feature_values = X[:, feature_index]
            paths = (paths << 1) | (feature_values > threshold)

        # Predict using the leaf values
        return np.array([tree.values[path] if path < len(tree.values) else 0 for path in paths])

