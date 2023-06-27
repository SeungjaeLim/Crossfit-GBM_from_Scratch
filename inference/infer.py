import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.metrics import mean_squared_error

def infer():
    model_fran = XGBoostScratch(n_estimators=50, learning_rate=0.1, reg_lambda=1, reg_alpha=1)
    model_helen = XGBoostScratch(n_estimators=50, learning_rate=0.1, reg_lambda=1, reg_alpha=1)
    model_grace = XGBoostScratch(n_estimators=50, learning_rate=0.1, reg_lambda=1, reg_alpha=1)
    model_filty50 = XGBoostScratch(n_estimators=50, learning_rate=0.1, reg_lambda=1, reg_alpha=1)
    model_fgonebad = XGBoostScratch(n_estimators=50, learning_rate=0.1, reg_lambda=1, reg_alpha=1)

    # Fit the model
    model_fran.fit(cf_datasets["fran"][0], cf_datasets["fran"][1])
    model_helen.fit(cf_datasets["helen"][0], cf_datasets["helen"][1])
    model_grace.fit(cf_datasets["grace"][0], cf_datasets["grace"][1])
    model_filty50.fit(cf_datasets["filthy50"][0], cf_datasets["filthy50"][1])
    model_fgonebad.fit(cf_datasets["fgonebad"][0], cf_datasets["fgonebad"][1])
    
    column_names = ['age', 'height', 'weight', 'run400', 'run5k', 'candj', 'snatch', 'deadlift', 'backsq', 'pullups', 'rec', 'high_school', 'college', 'pro', 'no_background', 'exp_coach', 'exp_alone', 'exp_courses', 'life_changing', 'exp_trainer', 'exp_level1', 'exp_start_nr', 'rest_plus', 'rest_minus', 'rest_sched', 'sched_0extra', 'sched_1extra', 'sched_2extra', 'sched_3extra', 'sched_nr', 'rest_nr', 'exp_1to2yrs', 'exp_2to4yrs', 'exp_4plus', 'exp_6to12mo', 'exp_lt6mo', 'eat_conv', 'eat_cheat', 'eat_quality', 'eat_paleo', 'eat_weigh', 'US', 'gender_']

    # Create an empty dictionary to store the column values
    column_values = {}

    # Iterate over the column names and ask for each column value
    for column in column_names:
        column_values[column] = input(f"Enter the value for column '{column}': ")

    # Create a DataFrame with the input data
    resdf = pd.DataFrame([column_values])
    resdf = resdf.astype(float)
    
    # Make predictions
    predictions_fran = model_fran.predict(resdf.to_numpy())
    predictions_helen = model_helen.predict(resdf.to_numpy())
    predictions_grace = model_grace.predict(resdf.to_numpy())
    predictions_filty50 = model_filty50.predict(resdf.to_numpy())
    predictions_fgonebad = model_fgonebad.predict(resdf.to_numpy())
    
    # Create a dictionary with the predicted values
    predictions = {
        'fran': predictions_fran[0],
        'helen': predictions_helen[0],
        'grace': predictions_grace[0],
        'filthy50': predictions_filty50[0],
        'fgonebad': predictions_fgonebad[0]
    }

    # Print the expected values with a prettier format
    for exercise, prediction in predictions.items():
        print(f"Expected {exercise}: {prediction:.2f}")
    
    return predictions
