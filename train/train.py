import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.metrics import mean_squared_error

def train():
    df = pd.read_csv('crossfit_games_athletes.csv')
    cf_datasets = preprocess(df)
    # List of models to evaluate
    models = [
        ('SimpleGradientBoost', SimpleGradientBoostScratch(n_estimators=50, learning_rate=0.1)),
        ('XGBoost', XGBoostScratch(n_estimators=50, learning_rate=0.1, reg_lambda=1, reg_alpha=1)),
        ('CatBoost', CatBoostScratch(n_estimators=50, learning_rate=0.1, depth=2)),
        ('LightGBM', LightGBMScratch(n_estimators=50, learning_rate=0.1, max_depth=2, n_bins=10))
    ]

    # Train and evaluate the models on each dataset
    for dataset_name, (X, y) in cf_datasets.items():
        # Split the data into training and test sets (30% for testing)
        split_idx = int(X.shape[0] * 0.7)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Iterate through each model
        for model_name, model in models:
            # Start the timer
            start_time = time.time()

            # Fit the model
            model.fit(X_train, y_train)

            # Calculate the fitting time
            fitting_time = time.time() - start_time

            # Print the training time
            print(f"{dataset_name} - {model_name}: Training time = {fitting_time:.2f} seconds")
            cf_fitting_times[f"t_{dataset_name}_{model_name}"] = fitting_time

            # Make predictions
            predictions = model.predict(X_test)

            # Evaluate the performance
            mse = mean_squared_error(y_test, predictions)
            # Save the MSE
            cf_mse_scores[f"mse_{dataset_name}_{model_name}"] = mse  # Save MSE to the dictionary

            # Display the results
            print(f"{dataset_name} - {model_name}: Mean Squared Error = {mse}")

            # Calculate errors between true target values and predictions
            errors = np.abs(y_test - predictions)

            # Get indices of the top 10 smallest errors
            top_10_indices = np.argsort(errors)[:10]

            for i, index in enumerate(top_10_indices):
                print(f"Index {index}: True value = {y_test[index]:.2f}, Prediction = {predictions[index]:.2f}, Error = {errors[index]:.2f}")

            # Plot comparison graph
            plt.figure(figsize=(10, 5))
            plt.bar(np.arange(10) - 0.2, y_test[top_10_indices], 0.4, label='True Values')
            plt.bar(np.arange(10) + 0.2, predictions[top_10_indices], 0.4, label='Predictions')
            plt.xlabel('Test Samples with Smallest Errors')
            plt.ylabel('Target Value')
            plt.title(f'True Values vs Predictions for {dataset_name} Dataset ({model_name})')
            plt.xticks(np.arange(10))
            plt.legend()
            plt.show()

            # Distribution Comparison
            plt.figure(figsize=(8, 4))
            plt.hist(y_test, bins=40, alpha=0.5, label='True Values')
            plt.hist(predictions, bins=40, alpha=0.5, label='Predictions')
            plt.xlabel('Target Value')
            plt.ylabel('Frequency')
            plt.title(f'Distribution of True Values vs Predictions for {dataset_name} Dataset ({model_name})')
            plt.legend()
            plt.show()

            # Add a separator for clarity
            print("\n" + "="*50 + "\n")
