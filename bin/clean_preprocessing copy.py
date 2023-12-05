import time
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

complete_data = pd.read_json('input/train.json')
data = pd.read_json('input/train.json')

# -------------------------------------------------Feature: entrytype-------------------------------------------------#
# It has 3 unique features
# So we one-hot encode the 'ENTRYTYPE' column
entrytype_dummies = pd.get_dummies(data['ENTRYTYPE'], prefix='entrytype')

# Drop the original 'ENTRYTYPE' column since it has been encoded
data.drop('ENTRYTYPE', axis=1, inplace=True)

# -------------------------------------------------Feature: publisher-------------------------------------------------#
# Impute the NA as 'unknown'
data['publisher'].fillna('Unknown', inplace=True)

# One-hot encode the 'publisher' column
publisher_dummies = pd.get_dummies(data['publisher'], prefix='publisher')
publisher_dummies

# Drop the original 'publisher' column since it has been encoded
data.drop('publisher', axis=1, inplace=True)

# -------------------------------------------------Feature: author-------------------------------------------------#
# Make it a set to get unique authors
all_authors = set()
for authors_list in data['author'].dropna():
    all_authors.update(authors_list)
    
# Count how many times each author appears
author_frequencies = Counter()

for authors_list in data['author'].dropna():
    author_frequencies.update(authors_list)

# Set Prolific Authors (After trials, decided on 25+ Publications)    
prolific_authors = [author for author, count in author_frequencies.items() if count >= 25]

author_dummies = data.copy()
author_dummies.info()

# One-hot encode these authors
for author in prolific_authors:
    author_dummies[f'author_{author}'] = author_dummies['author'].apply(lambda x: author in x if isinstance(x, list) else False)

# Make them into one variable: author_dummies
author_dummies.drop(['editor','title','year','author','abstract'], axis=1, inplace=True)
author_dummies = author_dummies.copy()

# Dropping Author from data since it has been encoded
data.drop('author', axis=1, inplace=True)
data.columns

# -------------------------------------------------New feature: author count-------------------------------------------------#

# Could be a good predictor: newer publications are more collaborative
author_count = complete_data['author'].apply(lambda x: len(x) if isinstance(x, list) else 0)


# FEATURE: TITLE

# Make Title Lower case
title_lower = data['title'].str.lower()

# Process the title with TF-IDF. Works better than hashing or count vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
title_tfidf = vectorizer.fit_transform(title_lower)

# Convert to DataFrame to be used in the prediction
title_tfidf_df = pd.DataFrame(title_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
title_processed = title_tfidf_df.copy()

# Drop title from data since it has been encoded
data.drop('title', axis=1, inplace=True)
data.columns


# FEATURE: ABSTRACT

# Make lowercase for further processing
abstract_lower = data['abstract'].fillna('no_abstract').str.lower()

# Abstract - COUNT VECTORIZER
abstract_vectorizer = CountVectorizer(stop_words='english', max_features=2000)
abstract_processed = abstract_vectorizer.fit_transform(abstract_lower)

# Convert to DataFrame to be used in the prediction
abstract_processed = pd.DataFrame(abstract_processed.toarray(), columns=abstract_vectorizer.get_feature_names_out())

# Drop abstract from data since it has been encoded
data.drop('abstract', axis=1, inplace=True)
data.columns


# NEW FEATURE: LENGTH OF ABSTRACT

abstract_length = abstract_lower.apply(len)
abstract_length


# FEATURE: EDITOR

# Replace missing values with with 'Unknown'
data['editor'].fillna('Unknown', inplace=True)

# Make a set of unique editors
all_editors = set()
for editors_list in data['editor']:
    if isinstance(editors_list, list):
        all_editors.update(editors_list)
    else:
        all_editors.add(editors_list)

# Calculate editor frequencies
editor_frequencies = Counter()
for editors_list in data['editor']:
    if isinstance(editors_list, list):
        editor_frequencies.update(editors_list)
    else:
        editor_frequencies.update([editors_list])

# One-hot encode editors which appears 5 or more times
threshold = 5
frequent_editors = [editor for editor, count in editor_frequencies.items() if count >= threshold]

editor_dummies = data.copy()
editor_dummies.info()

for editor in frequent_editors:
    editor_dummies[f'editor_{editor}'] = editor_dummies['editor'].apply(lambda x: editor in x if isinstance(x, list) else editor == x)

# Drop the original 'editor' column since it has been encoded
data.drop('editor', axis=1, inplace=True)
editor_dummies.drop('editor', axis=1, inplace=True)
editor_dummies.drop('year', axis=1, inplace=True)

# NEW FEATURE: NUMBER OF EDITORS

editor_count = complete_data['editor'].apply(lambda x: len(x) if isinstance(x, list) else 0)

###
###
###


# MODEL: RANDOM FOREST

X = pd.concat([entrytype_dummies, publisher_dummies, author_dummies, author_count, title_processed, abstract_processed, abstract_length, editor_dummies, editor_count], axis=1).copy()
y = data['year']

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initialize the Random Forest Regressor
model = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=0, max_depth=3)

# Start the training timer
train_start_time = time.time()

# Train the model
model.fit(X_train, y_train)

# Stop the training timer and print the time taken
train_end_time = time.time()
print(f"Training Time: {train_end_time - train_start_time} seconds")

# Start the prediction timer
predict_start_time = time.time()

# Predict on the testing set
y_pred = model.predict(X_test)

# Stop the prediction timer and print the time taken
predict_end_time = time.time()
print(f"Prediction Time: {predict_end_time - predict_start_time} seconds")

# Calculate Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")


# CROSS VALIDATION TO ENSURE NO OVERFITTING

model = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=0, max_depth=3)

# Define the number of folds for cross-validation
num_folds = 5

# Perform cross-validation and then calculate the mean absolute error for each fold
mae_scores = -cross_val_score(model, X, y, cv=num_folds, scoring='neg_mean_absolute_error')

# Calculate the average and standard deviation of the cross-validation MAE scores
average_mae = mae_scores.mean()
std_dev_mae = mae_scores.std()

print(f"Average MAE from cross-validation: {average_mae}")
print(f"Standard Deviation of MAE from cross-validation: {std_dev_mae}")