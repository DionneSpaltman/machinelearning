import pandas as pd
import logging
import json
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor  # Added import for RandomForest
from sklearn.metrics import mean_absolute_error

def main():
    start_time = time.time()  # Start the timer
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Loading training/test data")
    train = pd.DataFrame.from_records(json.load(open('train.json'))).fillna("")
    test = pd.DataFrame.from_records(json.load(open('test.json'))).fillna("")
    
    logging.info("Splitting validation")
    train, val = train_test_split(train, stratify=train['year'], random_state=123)

    # Updated featurizer to include 'abstract' using TfidfVectorizer
    featurizer = ColumnTransformer(
        transformers=[
            ("title_vec", CountVectorizer(), "title"),
            ("abstract_vec", TfidfVectorizer(), "abstract")  # New feature
        ],
        remainder='drop'
    )

    dummy = make_pipeline(featurizer, DummyRegressor(strategy='mean'))
    random_forest = make_pipeline(featurizer, RandomForestRegressor(n_estimators=100, n_jobs=-1))

    logging.info("Fitting models")
    dummy.fit(train.drop('year', axis=1), train['year'].values)
    random_forest.fit(train.drop('year', axis=1), train['year'].values)

    logging.info("Evaluating on validation data")
    dummy_err = mean_absolute_error(val['year'].values, dummy.predict(val.drop('year', axis=1)))
    logging.info(f"Mean baseline MAE: {dummy_err}")

    rf_err = mean_absolute_error(val['year'].values, random_forest.predict(val.drop('year', axis=1)))
    logging.info(f"Random Forest regressor MAE: {rf_err}")

    logging.info("Predicting on test")
    pred = random_forest.predict(test)
    test['year'] = pred

    logging.info("Writing prediction file")
    test.to_json("predicted.json", orient='records', indent=2)
    
    end_time = time.time()  # End the timer
    total_time = end_time - start_time
    logging.info(f"Total execution time: {total_time:.2f} seconds")
main()

# This is step 2 using random forest regressor, 
# Random Forest regressor MAE: 4.298209354954667
# but it's taking too long. 
# Total execution time: 968.14 seconds
# We'll do some optimization first in Step 3