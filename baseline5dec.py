import pandas as pd
import logging
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def main():
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Loading training/test data")
    train = pd.DataFrame.from_records(json.load(open('input/train.json'))).fillna("")
    test = pd.DataFrame.from_records(json.load(open('input/test.json'))).fillna("")
    logging.info("Splitting validation")
    
    # split into train and test 
    train, val = train_test_split(train, test_size=0.2, stratify=train['year'], random_state=123)

    # make author into strings 
    train["author"] = [','.join(map(str, i)) for i in train['author']]
    val["author"] = [','.join(map(str, i)) for i in val['author']]
    test["author"] = [','.join(map(str, i)) for i in test['author']]

    # handle missing values 
    train["abstract"] = train["abstract"].fillna("no abstract")
    train["editor"] = train["editor"].fillna("no editor")
    train["publisher"] = train["publisher"].fillna("no publisher")

    # one hot encode here
    train = pd.get_dummies(train,prefix=['ENTRYTYPE'], columns = ['ENTRYTYPE'])
    train = pd.get_dummies(train,prefix=['publisher'], columns = ['publisher'])

    featurizer = ColumnTransformer(
        transformers=[("title", CountVectorizer(), "title"), 
                        ("abstract", TfidfVectorizer(), "abstract"), 
                        ("author", CountVectorizer(), "author")
                      ],
        remainder='drop')
        
    dummy = make_pipeline(featurizer, DummyRegressor(strategy='mean'))
    random_forest = make_pipeline(featurizer, RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1))
    
    logging.info("Fitting models")
    dummy.fit(train.drop('year', axis=1), train['year'].values)
    random_forest.fit(train.drop('year', axis=1), train['year'].values)
    
    logging.info("Evaluating on validation data")
    err = mean_absolute_error(val['year'].values, dummy.predict(val.drop('year', axis=1)))
    
    logging.info(f"Mean baseline MAE: {err}")
    err = mean_absolute_error(val['year'].values, random_forest.predict(val.drop('year', axis=1)))
    
    logging.info(f"Random forest regress MAE: {err}")
    logging.info(f"Predicting on test")
    pred = random_forest.predict(test)
    test['year'] = pred
    
    logging.info("Writing prediction file")
    test.to_json("predicted5dec.json", orient='records', indent=2)
    
main()

