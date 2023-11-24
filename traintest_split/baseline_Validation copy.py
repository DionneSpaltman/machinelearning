# Hellooooooo
import pandas as pd
import logging
import json
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor 
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import time


def main():
    # Set logging to info 
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Loading training/test data")

    # Open test and training files 
    train = pd.DataFrame.from_records(json.load(open('train.json'))).fillna("")
    test = pd.DataFrame.from_records(json.load(open('test.json'))).fillna("")
    logging.info("Performing cross-validation")

    # Feature extraction for publisher and title 
    featurizer = ColumnTransformer(
        transformers=[("publisher", CountVectorizer(), "publisher"),
                      ("title", CountVectorizer(), "title")],
        remainder='drop')
    
    # Create a pipeline 
    dummy = make_pipeline(featurizer, DummyRegressor(strategy='mean'))
    ridge = make_pipeline(featurizer, Ridge())

    # Assuming 'year' is your target variable
    X = train.drop('year', axis=1)
    y = train['year'].values

    # Perform cross-validation
    cv_scores_dummy = cross_val_score(dummy, X, y, cv=5, scoring='neg_mean_absolute_error')
    cv_scores_ridge = cross_val_score(ridge, X, y, cv=20, scoring='neg_mean_absolute_error')
    # baseline without cv 4.8693545639728795
    # baseline with test_size at 0.2 = 4.857843171902147
    # baseline with test_size at 0.1 = 4.8350679150936875
    # baseline with test_size at 0.05 = 4.87466620970743
    # cv = 5 MAE: 4.850198258342784 +/- 0.03561417759235271
    # cv = 10 MAE: 4.81082260519578 +/- 0.05592474782778241
    # cv = 15 MAE: 4.80106158601259 +/- 0.05774546239386057
    # cv = 20 MAE: 4.79338459995168 +/- 0.07057959659685811


    # Display the cross-validation results
    logging.info(f'Mean baseline MAE: {-cv_scores_dummy.mean()} +/- {cv_scores_dummy.std()}')
    logging.info(f'Ridge MAE: {-cv_scores_ridge.mean()} +/- {cv_scores_ridge.std()}')

    logging.info("Fitting models")
    dummy.fit(X, y)
    ridge.fit(X, y)

    logging.info("Predicting on test")

    # Make predictions on the test data 
    pred_dummy = dummy.predict(test)
    test['year_dummy'] = pred_dummy

    pred_ridge = ridge.predict(test)
    test['year_ridge'] = pred_ridge

    logging.info("Writing prediction file")
    test.to_json("predictions/predicted_baseline.json", orient='records', indent=2)

main()

