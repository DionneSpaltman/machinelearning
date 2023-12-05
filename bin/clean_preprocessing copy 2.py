import time
import logging
import pandas as pd
import numpy as np
import json
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error


def main(): 
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Loading training/test data")
    train = pd.DataFrame.from_records(json.load(open('input/train.json'))).fillna("")
    test = pd.DataFrame.from_records(json.load(open('input/test.json'))).fillna("")
    logging.info("Splitting validation")

    train, val = train_test_split(train, stratify=train['year'], random_state=123)

    # -------------------------------------------------Feature: entrytype-------------------------------------------------#

    # One-hot encode 'ENTRYTYPE' and concatenate the result with the original DataFrame
    train = pd.concat([train, pd.get_dummies(train['ENTRYTYPE'], prefix='entrytype')], axis=1)

    # -------------------------------------------------Feature: publisher-------------------------------------------------#

    # Impute the NA as 'unknown'
    train['publisher'].fillna('Unknown', inplace=True)

    # One-hot encode the 'publisher' column
    train = pd.concat([train, pd.get_dummies(train['publisher'], prefix='publisher')], axis=1)

    # Drop the original 'publisher' column since it has been encoded > we don't want to drop it
    # train.drop('publisher', axis=1, inplace=True)

    # -------------------------------------------------Feature: author-------------------------------------------------#

    # Make it a set to get unique authors
    all_authors = set()
    for authors_list in train['author'].dropna():
        all_authors.update(authors_list)
        
    # Count how many times each author appears
    author_frequencies = Counter()
    for authors_list in train['author'].dropna():
        author_frequencies.update(authors_list)

    # Set Prolific Authors (After trials, decided on 25+ Publications)    
    prolific_authors = [author for author, count in author_frequencies.items() if count >= 25]

    author_dummies = train.copy()

    # One-hot encode these authors
    for author in prolific_authors:
        author_dummies[f'author_{author}'] = author_dummies['author'].apply(lambda x: author in x if isinstance(x, list) else False)

    # Make them into one variable: author_dummies
    author_dummies.drop(['editor','title','year','author','abstract'], axis=1, inplace=True)
    # author_dummies = author_dummies.copy()

    # One-hot encode these authors and add them to the 'train' DataFrame
    for author in prolific_authors:
        train[f'author_{author}'] = train['author'].apply(lambda x: author in x if isinstance(x, list) else False)

    # Dropping Author from data since it has been encoded > we can't drop it! 
    #train.drop('author', axis=1, inplace=True)
    #train.columns

    # -------------------------------------------------New feature: author count-------------------------------------------------#

    # Could be a good predictor: newer publications are more collaborative
    #author_count = train['author'].apply(lambda x: len(x) if isinstance(x, list) else 0)

    train['author_count'] = train['author'].apply(lambda x: len(x) if isinstance(x, list) else 0)


    # -------------------------------------------------Feature: title-------------------------------------------------#

    # Make Title Lower case
    title_lower = train['title'].str.lower()

    # Process the title with TF-IDF. Works better than hashing or count vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
    title_tfidf = vectorizer.fit_transform(title_lower)

    # Convert to DataFrame to be used in the prediction
    title_tfidf_df = pd.DataFrame(title_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
    # Concatenate 'title_tfidf_df' with the 'train' DataFrame
    train = pd.concat([train, title_tfidf_df], axis=1)


    # title_processed = title_tfidf_df.copy()

    # # Drop title from data since it has been encoded
    # train.drop('title', axis=1, inplace=True)
    # train.columns

    # -------------------------------------------------Feature: abstract-------------------------------------------------#

    # Make lowercase for further processing
    #abstract_lower = train['abstract'].fillna('no_abstract').str.lower()
    #abstract_lower = train['abstract'].fillna('no abstract').str.lower()

    # Abstract - COUNT VECTORIZER
    abstract_vectorizer = CountVectorizer(stop_words='english', max_features=2000)
    abstract_processed = abstract_vectorizer.fit_transform(train['abstract'])

    # Convert to DataFrame to be used in the prediction
    abstract_processed = pd.DataFrame(abstract_processed.toarray(), columns=abstract_vectorizer.get_feature_names_out())

    train = pd.concat([train, abstract_processed], axis=1)

    # # Drop abstract from training data since it has been encoded
    # train.drop('abstract', axis=1, inplace=True)
    # train.columns

    # -------------------------------------------------New feature: length of abstract-------------------------------------------------#

    #abstract_length = abstract_lower.apply(len)
    #abstract_length
    #abstract_length = pd.DataFrame(train['abstract'].apply(len))
    #train = pd.concat([train, abstract_length], axis=1, ignore_index=True)

    # CURRENTLY NOT USING THE LENFGTH OF THE ABSTRACT 
    # train['abstract_length'] = train['abstract'].apply(lambda x: len(x) if isinstance(x, (list, str)) else 0)


    # train['abstract_length'] = train['abstract'].apply(len)

    # -------------------------------------------------Feature: editor-------------------------------------------------#

    # Replace missing values with with 'Unknown'
    train['editor'].fillna('Unknown', inplace=True)

    # Make a set of unique editors
    all_editors = set()
    for editors_list in train['editor']:
        if isinstance(editors_list, list):
            all_editors.update(editors_list)
        else:
            all_editors.add(editors_list)

    # Calculate editor frequencies
    editor_frequencies = Counter()
    for editors_list in train['editor']:
        if isinstance(editors_list, list):
            editor_frequencies.update(editors_list)
        else:
            editor_frequencies.update([editors_list])

    # One-hot encode editors that appear 5 or more times
    threshold = 5
    frequent_editors = [editor for editor, count in editor_frequencies.items() if count >= threshold]

    #editor_dummies = train.copy()
    #editor_dummies.info()

    for editor in frequent_editors:
        #editor_dummies[f'editor_{editor}'] = editor_dummies['editor'].apply(lambda x: editor in x if isinstance(x, list) else editor == x)
        train[f'editor_{editor}'] = train['editor'].apply(lambda x: editor in x if isinstance(x, list) else editor == x)

    # Drop the original 'editor' column since it has been encoded
    #train.drop('editor', axis=1, inplace=True)
    #editor_dummies.drop('editor', axis=1, inplace=True)
    #editor_dummies.drop('year', axis=1, inplace=True)

    # -------------------------------------------------New feature: number of editors-------------------------------------------------#

    #editor_count = train['editor'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    train['editor_count'] = train['editor'].apply(lambda x: len(x) if isinstance(x, list) else 0)



    # -------------------------------------------------Model: Random Forest-------------------------------------------------#

    #X = pd.concat([entrytype_dummies, publisher_dummies, author_dummies, author_count, title_processed, abstract_processed, abstract_length, editor_dummies, editor_count], axis=1).copy()
    #y = train['year']

    featurizer = ColumnTransformer(
        transformers=[
            ("title", CountVectorizer(), "title")
            ("title_tfidf", TfidfVectorizer(), "title"),
            ("abstract_lower", "passthrough", ["abstract_lower"]),
            ("abstract_processed", CountVectorizer(), "abstract"),
            ("author_features", "passthrough", ["author_count"] + [f'author_{author}' for author in prolific_authors]),
            ("editor_features", "passthrough", ["editor_count"] + [f'editor_{editor}' for editor in frequent_editors])
            ],
        remainder='drop')


    random_forest = make_pipeline(featurizer, RandomForestRegressor())
    
    train, val = train_test_split(train, stratify=train['year'], test_size=0.2, random_state=0)

    logging.info("Fitting model")

    # Splitting the dataset
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Initialize the Random Forest Regressor
    #model = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=0, max_depth=3)

    # Start the training timer
    train_start_time = time.time()

    # Train the model
    # model.fit(X_train, y_train)
    random_forest.fit(train.drop('year', axis=1), train['year'].values)

    # Stop the training timer and print the time taken
    train_end_time = time.time()
    print(f"Training Time: {train_end_time - train_start_time} seconds")

    # Start the prediction timer
    predict_start_time = time.time()

    # Calculate Mean Absolute Error
    #mae = mean_absolute_error(y_test, y_pred)
    err = mean_absolute_error(val['year'].values, random_forest.predict(val.drop('year', axis=1)))
    print(f"Mean Absolute Error: {err}")

    # Predict on the testing set
    #y_pred = model.predict(X_test)
    #X_test['year'] = y_pred
    #X_test.drop('year', axis=1, inplace=True) 

    # Writing prediction file 
    test.to_json("predicted_2.json", orient='records', indent=2)

    # Stop the prediction timer and print the time taken
    predict_end_time = time.time()
    print(f"Prediction Time: {predict_end_time - predict_start_time} seconds")

    # -------------------------------------------------Cross-validation-------------------------------------------------#
"""

    # To ensure our model won't overfit
    model = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=0, max_depth=3)

    # Define the number of folds for cross-validation
    num_folds = 5

    # Perform cross-validation and then calculate the mean absolute error for each fold
    #mae_scores = -cross_val_score(model, X, y, cv=num_folds, scoring='neg_mean_absolute_error')
    err = mean_absolute_error(val['year'].values, random_forest.predict(val.drop('year', axis=1)))

    # Calculate the average and standard deviation of the cross-validation MAE scores
    average_mae = err.mean()
    std_dev_mae = err.std()

    print(f"Average MAE from cross-validation: {average_mae}")
    print(f"Standard Deviation of MAE from cross-validation: {std_dev_mae}")

"""

main()

