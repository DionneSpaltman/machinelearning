# doesn't work yet

import pandas as pd
import logging
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Perceptron
from sklearn.metrics import mean_absolute_error

def main():
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Loading training/test data")
    train = pd.DataFrame.from_records(json.load(open('train.json'))).fillna("")
    test = pd.DataFrame.from_records(json.load(open('test.json'))).fillna("")
    logging.info("Splitting validation")
    train, val = train_test_split(train, stratify=train['year'], random_state=123)
    
    # Extract column indices for the 'title' column
    title_col_index = train.columns.get_loc('title')
    
    featurizer = ColumnTransformer(
        transformers=[("title", CountVectorizer(), title_col_index)],
        remainder='drop')
    
    perceptron = make_pipeline(featurizer, Perceptron())
    logging.info("Fitting models")
    
    # Convert the output of CountVectorizer to dense array
    train_features = featurizer.fit_transform(train.drop('year', axis=1)).todense()
    perceptron.fit(train_features, train['year'].values)
    
    logging.info("Evaluating on validation data")
    
    # Convert the output of CountVectorizer to dense array for validation data
    val_features = featurizer.transform(val.drop('year', axis=1)).todense()
    err = mean_absolute_error(val['year'].values, perceptron.predict(val_features))
    
    logging.info(f"Perceptron MAE: {err}")
    logging.info(f"Predicting on test")
    
    # Convert the output of CountVectorizer to dense array for test data
    test_features = featurizer.transform(test).todense()
    pred = perceptron.predict(test_features)
    
    test['year'] = pred
    logging.info("Writing prediction file")
    test.to_json("predicted.json", orient='records', indent=2)

main()
