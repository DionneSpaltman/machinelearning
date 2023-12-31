"""
Baseline code but we improved it slightly
- Added comments to split up the code 
- Used the RandomForestRegressor instead of the Ridge Regressor 
- Can be called in main.py
- It works 
"""

import pandas as pd
import logging
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor 
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error


def baseline_improved(algorithm):
     # Set logging to info 
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Loading training/test data")

    # Open test and training files 
    train = pd.DataFrame.from_records(json.load(open('input/train.json'))).fillna("")
    test = pd.DataFrame.from_records(json.load(open('input/test.json'))).fillna("")
    logging.info("Splitting validation")

    # Split the training data into training data and validation data 
    train, val = train_test_split(train, stratify=train['year'], random_state=123)

    # Feature extraction for publisher and title 
    featurizer = ColumnTransformer(
        # transformers=[("publisher", CountVectorizer(), "publisher"),
        # ("title", CountVectorizer(), "title"), ("ENTRYTYPE", CountVectorizer(), "ENTRYTYPE")], 
        transformers=[("publisher", TfidfVectorizer(), "publisher"),
        ("title", TfidfVectorizer(), "title")], 
        remainder='drop')
    
    # Create a pipeline (ridge is default)
    dummy = make_pipeline(featurizer, DummyRegressor(strategy='mean'))
    if algorithm == "DecisionTreeRegressor":
        algorithm = make_pipeline(featurizer, DecisionTreeRegressor())
    elif algorithm == "RandomForestRegressor":
        algorithm = make_pipeline(featurizer, RandomForestRegressor(n_estimators=10, n_jobs=-1))
    else:
        algorithm = make_pipeline(featurizer, Ridge())
    
    logging.info("Decision")
    logging.info("Fitting models")
    dummy.fit(train.drop('year', axis=1), train['year'].values)
    algorithm.fit(train.drop('year', axis=1), train['year'].values)
    logging.info("Evaluating on validation data")

    # Evaluate the baseline model 
    err = mean_absolute_error(val['year'].values, dummy.predict(val.drop('year', axis=1)))
    logging.info(f"Mean baseline MAE: {err}")
    err = mean_absolute_error(val['year'].values, algorithm.predict(val.drop('year', axis=1)))
    logging.info(f"Ridge regress MAE: {err}")
    logging.info(f"Predicting on test")

    # Make predictions on the test data 
    pred = algorithm.predict(test)
    test['year'] = pred
    logging.info("Writing prediction file")
    test.to_json("predictions/predicted_baseline.json", orient='records', indent=2)
    

#main()

