import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_diabetes, fetch_california_housing
from sklearn.metrics import mean_squared_error

def preprocessor(df: pd.DataFrame):
    #removing problematic entries
    df = df[df['weight'] < 1500] #removes two anomolous weight entries of 1,750 and 2,113
    df = df[df['gender']!='--'] #removes 9 non-male/female gender entries due to small sample size
    df = df[df['age']>=18] #only considering adults
    df = df[(df['height']<96)&(df['height']>48)]#selects people between 4 and 8 feet

    #no lifts above world recording holding lifts were included
    df = df[(df['deadlift']>0)&(df['deadlift']<=1105)]
    df = df[(df['candj']>0)&(df['candj']<=395)]
    df = df[(df['snatch']>0)&(df['snatch']<=496)]
    df = df[(df['backsq']>0)&(df['backsq']<=1069)]
    df = df[(df['run400']>43)&(df['run400']<250)] #DNF
    df = df[(df['run5k']>756)&(df['run5k']<2500)] #DNF

    df = df[(df['fran']>1)&(df['fran']<1000)]
    df = df[(df['helen']>1)&(df['helen']<1500)]
    df = df[(df['grace']>1)&(df['grace']<1000)]
    df = df[(df['filthy50']>1)&(df['filthy50']<4000)] #DNF
    df = df[(df['fgonebad']>1)&(df['fgonebad']<1000)] #DNF

    #encoding background questions
    df['rec'] = np.where(df['background'].str.contains('I regularly play recreational sports'), 1, 0)
    df['high_school'] = np.where(df['background'].str.contains('I played youth or high school level sports'), 1, 0)
    df['college'] = np.where(df['background'].str.contains('I played college sports'), 1, 0)
    df['pro'] = np.where(df['background'].str.contains('I played professional sports'), 1, 0)
    df['no_background'] = np.where(df['background'].str.contains('I have no athletic background besides CrossFit'), 1, 0)

    #delete nonsense answers
    df = df[~(((df['high_school']==1)|(df['college']==1)|(df['pro']==1)|(df['rec']==1))&(df['no_background']==1))] #you can't have no background and also a background

    #create encoded columns for experience reponse
    df['exp_coach'] = np.where(df['experience'].str.contains('I began CrossFit with a coach'),1,0)
    df['exp_alone'] = np.where(df['experience'].str.contains('I began CrossFit by trying it alone'),1,0)
    df['exp_courses'] = np.where(df['experience'].str.contains('I have attended one or more specialty courses'),1,0)
    df['life_changing'] = np.where(df['experience'].str.contains('I have had a life changing experience due to CrossFit'),1,0)
    df['exp_trainer'] = np.where(df['experience'].str.contains('I train other people'),1,0)
    df['exp_level1'] = np.where(df['experience'].str.contains('I have completed the CrossFit Level 1 certificate course'),1,0)

    #delete nonsense answers
    df = df[~((df['exp_coach']==1)&(df['exp_alone']==1))] #you can't start alone and with a coach

    #creating no response option for coaching start
    df['exp_start_nr'] = np.where(((df['exp_coach']==0)&(df['exp_alone']==0)),1,0)

    #other options are assumed to be 0 if not explicitly selected

    #creating encoded columns with schedule data
    df['rest_plus'] = np.where(df['schedule'].str.contains('I typically rest 4 or more days per month'),1,0)
    df['rest_minus'] = np.where(df['schedule'].str.contains('I typically rest fewer than 4 days per month'),1,0)
    df['rest_sched'] = np.where(df['schedule'].str.contains('I strictly schedule my rest days'),1,0)

    df['sched_0extra'] = np.where(df['schedule'].str.contains('I usually only do 1 workout a day'),1,0)
    df['sched_1extra'] = np.where(df['schedule'].str.contains('I do multiple workouts in a day 1x a week'),1,0)
    df['sched_2extra'] = np.where(df['schedule'].str.contains('I do multiple workouts in a day 2x a week'),1,0)
    df['sched_3extra'] = np.where(df['schedule'].str.contains('I do multiple workouts in a day 3\+ times a week'),1,0)

    #removing/correcting problematic responses
    df = df[~((df['rest_plus']==1)&(df['rest_minus']==1))] #you can't have both more than and less than 4 rest days/month

    #points are only assigned for the highest extra workout value (3x only vs. 3x and 2x and 1x if multi selected)
    df['sched_0extra'] = np.where((df['sched_3extra']==1),0,df['sched_0extra'])
    df['sched_1extra'] = np.where((df['sched_3extra']==1),0,df['sched_1extra'])
    df['sched_2extra'] = np.where((df['sched_3extra']==1),0,df['sched_2extra'])
    df['sched_0extra'] = np.where((df['sched_2extra']==1),0,df['sched_0extra'])
    df['sched_1extra'] = np.where((df['sched_2extra']==1),0,df['sched_1extra'])
    df['sched_0extra'] = np.where((df['sched_1extra']==1),0,df['sched_0extra'])

    #adding no response columns
    df['sched_nr'] = np.where(((df['sched_0extra']==0)&(df['sched_1extra']==0)&(df['sched_2extra']==0)&(df['sched_3extra']==0)),1,0)
    df['rest_nr'] = np.where(((df['rest_plus']==0)&(df['rest_minus']==0)),1,0)

    # encoding howlong (crossfit lifetime)
    df['exp_1to2yrs'] = np.where((df['howlong'].str.contains('1-2 years')),1,0)
    df['exp_2to4yrs'] = np.where((df['howlong'].str.contains('2-4 years')),1,0)
    df['exp_4plus'] = np.where((df['howlong'].str.contains('4\+ years')),1,0)
    df['exp_6to12mo'] = np.where((df['howlong'].str.contains('6-12 months')),1,0)
    df['exp_lt6mo'] = np.where((df['howlong'].str.contains('Less than 6 months')),1,0)

    #keeping only higest repsonse
    df['exp_lt6mo'] = np.where((df['exp_4plus']==1),0,df['exp_lt6mo'])
    df['exp_6to12mo'] = np.where((df['exp_4plus']==1),0,df['exp_6to12mo'])
    df['exp_1to2yrs'] = np.where((df['exp_4plus']==1),0,df['exp_1to2yrs'])
    df['exp_2to4yrs'] = np.where((df['exp_4plus']==1),0,df['exp_2to4yrs'])
    df['exp_lt6mo'] = np.where((df['exp_2to4yrs']==1),0,df['exp_lt6mo'])
    df['exp_6to12mo'] = np.where((df['exp_2to4yrs']==1),0,df['exp_6to12mo'])
    df['exp_1to2yrs'] = np.where((df['exp_2to4yrs']==1),0,df['exp_1to2yrs'])
    df['exp_lt6mo'] = np.where((df['exp_1to2yrs']==1),0,df['exp_lt6mo'])
    df['exp_6to12mo'] = np.where((df['exp_1to2yrs']==1),0,df['exp_6to12mo'])
    df['exp_lt6mo'] = np.where((df['exp_6to12mo']==1),0,df['exp_lt6mo'])

    #encoding dietary preferences
    df['eat_conv'] = np.where((df['eat'].str.contains('I eat whatever is convenient')),1,0)
    df['eat_cheat']= np.where((df['eat'].str.contains('I eat 1-3 full cheat meals per week')),1,0)
    df['eat_quality']= np.where((df['eat'].str.contains('I eat quality foods but don\'t measure the amount')),1,0)
    df['eat_paleo']= np.where((df['eat'].str.contains('I eat strict Paleo')),1,0)
    df['eat_cheat']= np.where((df['eat'].str.contains('I eat 1-3 full cheat meals per week')),1,0)
    df['eat_weigh'] = np.where((df['eat'].str.contains('I weigh and measure my food')),1,0)

    #encoding location as US vs non-US
    US_regions = ['Southern California', 'North East', 'North Central','South East', 'South Central', 'South West', 'Mid Atlantic','Northern California','Central East', 'North West']
    df['US'] = np.where((df['region'].isin(US_regions)),1,0)

    #encoding gender
    df['gender_'] = np.where(df['gender']=='Male',1,0)

    # Boolean mask initialized to False for all rows
    outliers_mask = pd.Series([False]*len(df))

    columns_to_check = ['fran', 'helen', 'grace', 'filthy50', 'fgonebad', 'run400', 'run5k', 'pullups']

    # Updating the mask for each column
    for column in columns_to_check:
        median = df[column].median()
        sigma = df[column].std()
        outliers_mask = outliers_mask | (df[column] > (median + 2 * sigma))

    # Filtering the DataFrame by negating the mask (keeping the rows that are NOT outliers)
    df = df[~outliers_mask]

    # Specify target columns
    target_columns = ['fran', 'helen', 'grace', 'filthy50', 'fgonebad']

    # Separating the features and the targets
    df_X = df.drop(target_columns, axis=1)
    df_y = df[target_columns]

    # Define the datasets and their names
    cf_datasets = {
        "fran": (df_X.to_numpy(), df_y['fran'].to_numpy()),
        "helen": (df_X.to_numpy(), df_y['helen'].to_numpy()),
        "grace": (df_X.to_numpy(), df_y['grace'].to_numpy()),
        "filthy50": (df_X.to_numpy(), df_y['filthy50'].to_numpy()),
        "fgonebad": (df_X.to_numpy(), df_y['fgonebad'].to_numpy()),
    }

    return cf_datasets
