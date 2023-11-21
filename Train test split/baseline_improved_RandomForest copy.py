import pandas as pd
import logging
import json
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import mean_absolute_error

def main():
    # Set logging to info 
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Loading training/test data")

    # Open test and training files 
    train = pd.DataFrame.from_records(json.load(open('train.json'))).fillna("")
    test = pd.DataFrame.from_records(json.load(open('test.json'))).fillna("")
    logging.info("Splitting validation")

    # Feature extraction for publisher and title 
    featurizer = ColumnTransformer(
        transformers=[("publisher", CountVectorizer(), "publisher"),
                      ("title", CountVectorizer(), "title")], 
        remainder='drop')
    
    # Create a pipeline 
 #   dummy = make_pipeline(featurizer, DummyRegressor(strategy='mean'))
    random_forest = make_pipeline(featurizer, RandomForestRegressor(n_estimators=10, n_jobs=-1))

    # Assuming 'year' is your target variable
    X = train.drop('year', axis=1)
    y = train['year'].values

    # Perform cross-validation
#    cv_scores_dummy = cross_val_score(dummy, X, y, cv=5, scoring='neg_mean_absolute_error')
    cv_scores_random_forest = cross_val_score(random_forest, X, y, cv=15, scoring='neg_mean_absolute_error')
    # baseline Random Forest MAE: 4.129321705298972
    # test_size = 0.2 Random Forest MAE: 4.116458653465728
    # test_size = 0.1 Random Forest MAE: 4.194229718385932
    # cv = 5 MAE: 4.179684370625555 +/- 0.044847389438292314
    # cv = 10 MAE: 4.182240335808242 +/- 0.04496826144569115
    # cv = 15 MAE: 4.150541293034312 +/- 0.07283829479707211

    # Display the cross-validation results
 #   logging.info(f'Mean baseline MAE: {-cv_scores_dummy.mean()} +/- {cv_scores_dummy.std()}')
    logging.info(f'Random Forest MAE: {-cv_scores_random_forest.mean()} +/- {cv_scores_random_forest.std()}')

    logging.info("Fitting models")
#   dummy.fit(X, y)
    random_forest.fit(X, y)

    logging.info("Predicting on test")

    # Make predictions on the test data 
#    pred_dummy = dummy.predict(test)
#    test['year_dummy'] = pred_dummy

    pred_random_forest = random_forest.predict(test)
    test['year_random_forest'] = pred_random_forest

    logging.info("Writing prediction file")
    test.to_json("predictions/predicted_baseline.json", orient='records', indent=2)

main()
