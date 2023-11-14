import pandas as pd
import logging
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer  # Change to TfidfVectorizer
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
    
    featurizer = ColumnTransformer(
        transformers=[("title", TfidfVectorizer(), "title")],  # Change to TfidfVectorizer
        remainder='drop')
    
    # Increase max_iter for faster convergence
    perceptron = make_pipeline(featurizer, Perceptron(max_iter=10000))  # Experiment with max_iter
    
    logging.info("Fitting models")
    perceptron.fit(train.drop('year', axis=1), train['year'].values)
    
    logging.info("Evaluating on validation data")
    err = mean_absolute_error(val['year'].values, perceptron.predict(val.drop('year', axis=1)))
    logging.info(f"Perceptron MAE: {err}")

    logging.info("Predicting on test")
    pred = perceptron.predict(test)
    test['year'] = pred
    logging.info("Writing prediction file")
    test.to_json("predicted.json", orient='records', indent=2)
    
main()
