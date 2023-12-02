import pandas as pd
import logging
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
from collections import Counter

def main():
    logging.getLogger().setLevel(logging.INFO)
    logging.info("Loading training/test data")
    train = pd.DataFrame.from_records(json.load(open('input/train.json'))).fillna("")
    test = pd.DataFrame.from_records(json.load(open('input/test.json'))).fillna("")

    # -------------------------------------------------Feature: entrytype-------------------------------------------------#
    entrytype_dummies = pd.get_dummies(train['ENTRYTYPE'], prefix='entrytype')
    train.drop('ENTRYTYPE', axis=1, inplace=True)

    # -------------------------------------------------Feature: publisher-------------------------------------------------#
    train['publisher'].fillna('Unknown', inplace=True)
    publisher_dummies = pd.get_dummies(train['publisher'], prefix='publisher')

    # -------------------------------------------------Feature: author-------------------------------------------------#
    all_authors = set()
    for authors_list in train['author'].dropna():
        all_authors.update(authors_list)
        
    author_frequencies = Counter()
    for authors_list in train['author'].dropna():
        author_frequencies.update(authors_list)

    prolific_authors = [author for author, count in author_frequencies.items() if count >= 25]
    author_dummies = train.copy()
    for author in prolific_authors:
        author_dummies[f'author_{author}'] = author_dummies['author'].apply(lambda x: author in x if isinstance(x, list) else False)
    author_dummies.drop(['editor','title','year','author','abstract'], axis=1, inplace=True)

    # -------------------------------------------------New feature: author count-------------------------------------------------#
    author_count = train['author'].apply(lambda x: len(x) if isinstance(x, list) else 0)

    # -------------------------------------------------Feature: title-------------------------------------------------#
    title_lower = train['title'].str.lower()
    vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
    title_tfidf = vectorizer.fit_transform(title_lower)
    title_processed = pd.DataFrame(title_tfidf.toarray(), columns=vectorizer.get_feature_names_out())

    # -------------------------------------------------Feature: abstract-------------------------------------------------#
    abstract_lower = train['abstract'].fillna('no_abstract').str.lower()
    abstract_vectorizer = CountVectorizer(stop_words='english', max_features=2000)
    abstract_processed = pd.DataFrame(abstract_vectorizer.fit_transform(abstract_lower).toarray(), columns=abstract_vectorizer.get_feature_names_out())

    # -------------------------------------------------New feature: length of abstract-------------------------------------------------#
    abstract_length = abstract_lower.apply(len)

    # -------------------------------------------------Feature: editor-------------------------------------------------#
    train['editor'].fillna('Unknown', inplace=True)
    all_editors = set()
    for editors_list in train['editor']:
        if isinstance(editors_list, list):
            all_editors.update(editors_list)
        else:
            all_editors.add(editors_list)

    editor_frequencies = Counter()
    for editors_list in train['editor']:
        if isinstance(editors_list, list):
            editor_frequencies.update(editors_list)
        else:
            editor_frequencies.update([editors_list])

    frequent_editors = [editor for editor, count in editor_frequencies.items() if count >= 5]
    editor_dummies = train.copy()
    for editor in frequent_editors:
        editor_dummies[f'editor_{editor}'] = editor_dummies['editor'].apply(lambda x: editor in x if isinstance(x, list) else editor == x)
    editor_dummies.drop(['editor', 'year'], axis=1, inplace=True)

    # -------------------------------------------------New feature: number of editors-------------------------------------------------#
    editor_count = train['editor'].apply(lambda x: len(x) if isinstance(x, list) else 0)

    # -------------------------------------------------Model: Random Forest-------------------------------------------------#
    X = pd.concat([entrytype_dummies, publisher_dummies, author_dummies, author_count, title_processed, abstract_processed, abstract_length, editor_dummies, editor_count], axis=1)
    y = train['year']

    # Splitting the dataset
    train, val = train_test_split(train, stratify=train['year'], test_size=0.2, random_state=0)

    # Initialize and fit the Random Forest Regressor
    model = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=0, max_depth=3)
    model.fit(train.drop('year', axis=1), train['year'].values)

    # Evaluate on validation data
    val_pred = model.predict(val.drop('year', axis=1))
    err = mean_absolute_error(val['year'].values, val_pred)
    logging.info(f"Random Forest MAE: {err}")

    # Predict on test data
    test_pred = model.predict(test)
    test['year'] = test_pred

    # Write prediction file
    test.to_json("predicted2.json", orient='records', indent=2)

main()
