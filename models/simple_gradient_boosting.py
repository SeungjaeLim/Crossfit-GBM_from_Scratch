import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.metrics import mean_squared_error

class DecisionTreeStump:
    """A decision tree stump model.

    Attributes:
        feature_index (int): The index of the feature used for splitting.
        threshold (float): The threshold value for splitting.
        left_value (float): The value assigned to the left side of the split.
        right_value (float): The value assigned to the right side of the split.
    """

    def __init__(self):
        """Initialize a decision tree stump with default attribute values."""
        self.feature_index = 0
        self.threshold = 0
        self.left_value = 0
        self.right_value = 0
        
class SimpleGradientBoostScratch:
    """A simple implementation of the Gradient Boosting algorithm.

    Parameters:
        n_estimators (int): The number of boosting iterations (default: 100).
        learning_rate (float): The learning rate or shrinkage factor (default: 0.1).

    Attributes:
        n_estimators (int): The number of boosting iterations.
        learning_rate (float): The learning rate or shrinkage factor.
        trees (list): The list of decision tree stumps in the ensemble.
        training_errors (list): The training errors (mean squared error) at each iteration.
        test_errors (list): The test errors (mean squared error) at each iteration.
        init_prediction (float): The initial prediction made by the ensemble.
    """

    def __init__(self, n_estimators=100, learning_rate=0.1):
        """Initialize a SimpleGradientBoost object with the specified parameters.

        Args:
            n_estimators (int, optional): The number of boosting iterations (default: 100).
            learning_rate (float, optional): The learning rate or shrinkage factor (default: 0.1).
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.trees = []
        self.training_errors = []
        self.test_errors = []
        self.init_prediction = 0

    def fit(self, X, y, X_test=None, y_test=None):
        """Fit the boosting model to the given training data.

        Args:
            X (array-like): The input features of the training data.
            y (array-like): The target values of the training data.
            X_test (array-like, optional): The input features of the test data (default: None).
            y_test (array-like, optional): The target values of the test data (default: None).
        """
        # Initialize model with a mean prediction
        self.init_prediction = np.mean(y)
        pred = np.full_like(y, self.init_prediction)
        residuals = y - pred

        for _ in range(self.n_estimators):
            # Fit a decision tree to the residuals
            tree = self.fit_stump(X, residuals)
            self.trees.append(tree)

            # Update predictions
            pred += self.learning_rate * self.predict_stump(X, tree)

            # Recompute residuals
            residuals = y - pred

            # Store the training error (mean squared error)
            self.training_errors.append(np.mean(residuals ** 2))

            if X_test is not None and y_test is not None:
                test_pred = self.predict(X_test)
                test_residuals = y_test - test_pred
                self.test_errors.append(np.mean(test_residuals ** 2))

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
        """Fit a decision tree stump (depth-1 decision tree) to the given data.

        Args:
            X (array-like): The input features of the data.
            y (array-like): The target values of the data.

        Returns:
            DecisionTreeStump: The fitted decision tree stump.
        """
        # Fit a decision tree stump (depth-1 decision tree) to the residuals
        n_samples, n_features = X.shape
        best_feature = None
        best_threshold = None
        best_left_value = None
        best_right_value = None
        best_error = np.inf

        for feature_index in range(n_features):
            feature_values = X[:, feature_index]
            for threshold in feature_values:
                # Left and right splits
                left_mask = feature_values <= threshold
                right_mask = ~left_mask

                # Calculate values in the leaves as an average of residuals
                left_value = np.mean(y[left_mask]) if np.any(left_mask) else 0
                right_value = np.mean(y[right_mask]) if np.any(right_mask) else 0

                # Calculate mean squared error
                residuals = np.where(left_mask, left_value, right_value)
                error = np.sum((y - residuals) ** 2)

                # Keep track of the best split
                if error < best_error:
                    best_feature = feature_index
                    best_threshold = threshold
                    best_left_value = left_value
                    best_right_value = right_value
                    best_error = error

        # Store the best split in the decision tree stump
        stump = DecisionTreeStump()
        stump.feature_index = best_feature
        stump.threshold = best_threshold
        stump.left_value = best_left_value
        stump.right_value = best_right_value
        return stump

    def predict_stump(self, X, tree):
        """Make predictions using a decision tree stump.

        Args:
            X (array-like): The input features for making predictions.
            tree (DecisionTreeStump): The decision tree stump.

        Returns:
            array-like: The predicted target values.
        """
        return np.where(X[:, tree.feature_index] <= tree.threshold, tree.left_value, tree.right_value)
