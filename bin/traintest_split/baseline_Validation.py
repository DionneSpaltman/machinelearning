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
    logging.info("Splitting validation")

    # Split the training data into training data and validation data 
    train, val = train_test_split(train, stratify=train['year'], test_size = 0.05, random_state=123)

    # Feature extraction for publisher and title 
    featurizer = ColumnTransformer(
        # transformers=[("publisher", CountVectorizer(), "publisher"),
        # ("title", CountVectorizer(), "title"), ("ENTRYTYPE", CountVectorizer(), "ENTRYTYPE")], 
        transformers=[("publisher", CountVectorizer(), "publisher"),
        ("title", CountVectorizer(), "title")], 
        remainder='drop')
    
    # Create a pipeline 
    dummy = make_pipeline(featurizer, DummyRegressor(strategy='mean'))
 #   ridge = make_pipeline(featurizer, DecisionTreeRegressor())
    ridge = make_pipeline(featurizer, Ridge())

    train_start_time = time.time()
    logging.info("Decision")
    logging.info("Fitting models")
    dummy.fit(train.drop('year', axis=1), train['year'].values)
    ridge.fit(train.drop('year', axis=1), train['year'].values)
    logging.info("Evaluating on validation data")

    # Evaluate the baseline model 
    err = mean_absolute_error(val['year'].values, dummy.predict(val.drop('year', axis=1)))
    logging.info(f"Mean baseline MAE: {err}")
    err = mean_absolute_error(val['year'].values, ridge.predict(val.drop('year', axis=1)))
    logging.info(f"Random Forest MAE: {err}")
    logging.info(f"Predicting on test")

    # Make predictions on the test data 
    pred = ridge.predict(test)
    test['year'] = pred
    logging.info("Writing prediction file")
    test.to_json("predictions/predicted_baseline.json", orient='records', indent=2)
    train_end_time = time.time()
    print(f"Training Time: {train_end_time - train_start_time} seconds")
    
main()

