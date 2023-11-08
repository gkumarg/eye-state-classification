'''Train the final model for Eye-State project'''

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.io.arff import loadarff  # To load arff files
from scipy import stats

from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from xgboost import XGBClassifier

import pickle


DATA_DIR = Path('../data')
MODEL_DIR = Path('../models')

def load_data():
    # read arff data from data folder
    data, meta = loadarff(DATA_DIR / 'EEG Eye State.arff')
    df = pd.DataFrame(data.tolist(), columns=meta.names())
    # Convert the label to int
    df['eyeDetection'] = df['eyeDetection'].str.decode('utf-8').astype(int)
    return df

def create_train_test_splits(df, test_len=0.2):
    test_size = int(len(df) * test_len)
    train_size = len(df) - test_size

    train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
    print(f'Train: {len(train)}')
    print(f'Test: {len(test)}')
    return train, test

def remove_outliers(df, feature):
    # find outliers using z-score and fill with nan, then impute with linear interpolation for each feature and print how many records were updated
    z_scores = stats.zscore(df[feature])
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3)
    # count how many records were updated
    print(f'Number of outliers in {feature}: {len(df[~filtered_entries])}')
    df.loc[~filtered_entries, feature] = np.nan
    df[feature] = df[feature].interpolate(method='linear')
    return df


def get_X_y(df):

    X = df.drop('eyeDetection', axis=1)
    y = df['eyeDetection']

    return X, y

def cross_val_model(model, X, y, tscv):

    # evaluate the model
    scores = cross_validate(model, X, y, cv=tscv, scoring=['accuracy', 'precision', 'recall', 'f1'] )    
    return scores

def tune_model(model, X, y, tscv):
    # define the grid
    param_grid = {
        'max_depth': [3, 4, 5, 6, 7],
        'n_estimators': [100, 200, 300, 400, 500],
        'learning_rate': [0.01, 0.1, 0.2, 0.3]
    }

    # define the grid search
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='f1', n_jobs=-1, cv=tscv, verbose=2)

    # fit the grid search
    grid_result = grid.fit(X, y)

    # summarize results
    print(f'Best: {grid_result.best_score_} using {grid_result.best_params_}')
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    # save the results
    results = pd.DataFrame({'mean': means, 'std': stds, 'params': params})
    # results.to_csv(DATA_DIR / 'xgb_grid_search_results.csv', index=False)

    return grid_result.best_estimator_, results

def train_xgb(model, X, y):
    
    # fit the model
    model.fit(X, y)

    return model


def save_model(fitted_model):

    # save the model to disk    
    MODEL_DIR.mkdir(exist_ok=True)

    # save the model to disk
    filename = 'xgb_model.bin'
    pickle.dump(fitted_model, open(MODEL_DIR / filename, 'wb'))
    print("Saved model to disk")

if __name__ == '__main__':

    # load the data
    df = load_data()

    # create train and test splits
    train, test = create_train_test_splits(df)
    
    for feature in df.columns:
        if feature != 'eyeDetection':
            train_df = remove_outliers(train, feature)
            test_df = remove_outliers(test, feature)

    X, y = get_X_y(train_df)
    X_test, y_test = get_X_y(test_df)


    # tune the model
    model = XGBClassifier()
    # perform time series split for validation
    tscv = TimeSeriesSplit(n_splits=3, test_size=300)
    best_model, results = tune_model(model, X, y, tscv)

    print(f"Tuning parameters and score: {results}")
    print(f"Best Model after tuning: {best_model}")
    

    # cross validate the model
    scores = cross_val_model(best_model, X, y, tscv)
    # print(scores)
    print(f'Test Accuracy: {scores["test_accuracy"].mean()*100:0.2f}%')
    print(f'Test Precision: {scores["test_precision"].mean()*100:0.2f}%')
    print(f'Test Recall: {scores["test_recall"].mean()*100:0.2f}%')
    print(f'Test F1: {scores["test_f1"].mean()*100:0.2f}%')

    # train the model
    fitted_model = train_xgb(best_model, X, y)

    print(f"Final fitted Model: {fitted_model}")
    
    # save the model
    save_model(fitted_model)

    # make predictions for test data
    y_pred = fitted_model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))

    print("Done")



