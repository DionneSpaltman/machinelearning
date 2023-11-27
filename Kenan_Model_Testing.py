"""
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA

# WHOLE DATA

data = pd.read_json('input/train.json')
print(data.head())

# Basic info about data
print(data.info())

# Summary statistics for numerical fields
print(data.describe())


# YEAR COLUMN


# Distribution of the 'year' field
print(data['year'].value_counts())


# ENTRYTYPE COLUMN


print(data['ENTRYTYPE'].unique())
# Only 3 types, probably good to use One-Hot Encoding

# One-hot encode the 'ENTRYTYPE' column
entrytype_dummies = pd.get_dummies(data['ENTRYTYPE'], prefix='entrytype')

# Concatenate the new columns with the original dataframe
data = pd.concat([data, entrytype_dummies], axis=1)

# Drop the original 'ENTRYTYPE' column
data.drop('ENTRYTYPE', axis=1, inplace=True)
print(data.info())


# EDITOR COLUMN


# Count null values in the 'editor' column
null_count = data['editor'].isnull().sum()
print(f"Number of null values in 'editor': {null_count}")
# 64438 null values from 65914 data points. I guess.. Delete it?

data.drop('editor', axis=1, inplace=True)
print(data.info())


# PUBLISHER COLUMN


# Display unique values in the 'publisher' column
unique_publishers = data['publisher'].unique()
print(unique_publishers[:30])  # Display the first 30 unique values

# Count of unique values
print(f"Number of unique publishers: {data['publisher'].nunique()}")
# There are 120 Unique Publishers

# Display the frequency of each publisher
publisher_counts = data['publisher'].value_counts()
print(publisher_counts.head(20))  # Display the top 20 most frequent publishers

# Count null values in the 'publisher' column
null_count = data['publisher'].isnull().sum()
print(f"Number of null values in 'publisher': {null_count}")
# 8201 null values from 65914 Data Points

# Relationship Between Publisher and Year
# Calculate mean year for each publisher
mean_years = data.groupby('publisher')['year'].mean().sort_values()
mean_years


# Impute the NA as 'unknown_publisher'

data['publisher'].fillna('Unknown', inplace=True)

# One-hot encoding of the 'publisher' column
publisher_dummies = pd.get_dummies(data['publisher'], prefix='publisher')

# Join the one-hot encoded columns back to the original DataFrame
data = pd.concat([data, publisher_dummies], axis=1)


# AUTHOR COLUMN


# Flatten the list of authors
all_authors = set()
for authors_list in data['author'].dropna():
    all_authors.update(authors_list)

# Now all_authors contains all unique authors

# Initialize a Counter object to hold author frequencies
author_frequencies = Counter()

# Iterate through the author lists and update the counter
for authors_list in data['author'].dropna():
    author_frequencies.update(authors_list)

# Now author_frequencies contains the count of each author
author_frequencies

# Assuming each entry in the 'author' column is a list of authors
data['author_count'] = data['author'].apply(lambda x: len(x) if isinstance(x, list) else 0)

# Identify authors with 50+ publications
prolific_authors = [author for author, count in author_frequencies.items() if count >= 50]

# One-hot encode these authors
for author in prolific_authors:
    data[f'author_{author}'] = data['author'].apply(lambda x: author in x if isinstance(x, list) else False)

prolific_authors

# TITLE AND ABSTRACT COLUMN 

# Make Title Lower case
data['title_processed'] = data['title'].str.lower()

# Feature Extraction: TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=500)  # Limit features to 500
title_tfidf = vectorizer.fit_transform(data['title_processed'])

# Convert to DataFrame
title_tfidf_df = pd.DataFrame(title_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
title_tfidf_df

# Lowercase Abstract
data['abstract_processed'] = data['abstract'].fillna('').str.lower()

# Feature Extraction: TF-IDF for 'abstract'
abstract_vectorizer = TfidfVectorizer(stop_words='english', max_features=500)  # Limit features to 500
abstract_tfidf = abstract_vectorizer.fit_transform(data['abstract_processed'])

# Convert 'abstract' TF-IDF to DataFrame
abstract_tfidf_df = pd.DataFrame(abstract_tfidf.toarray(), columns=abstract_vectorizer.get_feature_names_out())

data.info()

'''
# RANDOM FOREST MODEL PART 1


X = pd.concat([data.drop(['year', 'title', 'abstract', 'publisher', 'author', 'title_processed', 'abstract_processed'], axis=1),
               title_tfidf_df, abstract_tfidf_df], axis=1).copy()
y = data['year']

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initialize the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=0)

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

# Time taken: 6 minutes
# MAE: 3.47


# FEATURE IMPORTANCE ANALYSIS


# Extracting feature importances
feature_importances = model.feature_importances_

# Matching feature names with their importances
feature_names = X_train.columns
importances = pd.Series(feature_importances, index=feature_names)

# Sorting the features by their importance
sorted_importances = importances.sort_values(ascending=False)

# Visualizing the top 20 most important features
plt.figure(figsize=(15, 3))
sorted_importances[:20].plot(kind='bar')
plt.title('Top 20 Feature Importances in Random Forest Model')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()

# The most important feature is "Publisher_Unknown" 
# (means we should not delete the missing publisher instances)
# Maybe if we impute it somehow, the prediction will be even better?
# Author count is also important. Seems like in the more recent years,
# people collaborate more (number of writers goes up)
'''

# NEW FEATURE ENGINEERING METHODS

'''
# AUTHOR COLUMN. Let's make top authors someone with 100+ publications instead


# Flatten the list of authors
all_authors = set()
for authors_list in data['author'].dropna():
    all_authors.update(authors_list)

# Now all_authors contains all unique authors

# Initialize a Counter object to hold author frequencies
author_frequencies = Counter()

# Iterate through the author lists and update the counter
for authors_list in data['author'].dropna():
    author_frequencies.update(authors_list)

# Now author_frequencies contains the count of each author
author_frequencies

# Assuming each entry in the 'author' column is a list of authors
data['author_count'] = data['author'].apply(lambda x: len(x) if isinstance(x, list) else 0)

# Identify authors with 50+ publications
prolific_authors = [author for author, count in author_frequencies.items() if count >= 100]

# One-hot encode these authors
for author in prolific_authors:
    data[f'author_{author}'] = data['author'].apply(lambda x: author in x if isinstance(x, list) else False)

prolific_authors

'''

# TITLE AND ABSTRACT COLUMN
# Limit Features to 200

'''
# Make Title Lower case
data['title_processed'] = data['title'].str.lower()

# Feature Extraction: TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=200)  # Limit features to 200
title_tfidf = vectorizer.fit_transform(data['title_processed'])

# Convert to DataFrame
title_tfidf_df = pd.DataFrame(title_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
title_tfidf_df

# Lowercase Abstract
data['abstract_processed'] = data['abstract'].fillna('').str.lower()

# Feature Extraction: TF-IDF for 'abstract'
abstract_vectorizer = TfidfVectorizer(stop_words='english', max_features=200)  # Limit features to 200
abstract_tfidf = abstract_vectorizer.fit_transform(data['abstract_processed'])

# Convert 'abstract' TF-IDF to DataFrame
abstract_tfidf_df = pd.DataFrame(abstract_tfidf.toarray(), columns=abstract_vectorizer.get_feature_names_out())

data.info()

'''
# ADDING INTERACTION TERM FOR AUTHOR COUNT TIMES ENTRYTYPES

# Assuming 'author_count' and 'entrytype_article' are columns in your data
data['author_article_interaction'] = data['author_count'] * data['entrytype_article']
data['author_inproceedings_interaction'] = data['author_count'] * data['entrytype_inproceedings']
data['author_proceedings_interaction'] = data['author_count'] * data['entrytype_proceedings']


# TRYING ADVANCE TEXT ANALYSIS ON THE ABSTRACT

# Prepare a CountVectorizer, LDA expects integer counts
count_vect = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
doc_term_matrix = count_vect.fit_transform(data['abstract_processed'])

# Number of topics
n_topics = 10  # Adjust based on your needs

# Create and fit LDA model
lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
lda.fit(doc_term_matrix)

# Example: print the top words for each topic
words = count_vect.get_feature_names()
for i, topic in enumerate(lda.components_):
    print(f"Top words for topic {i}:")
    print([words[i] for i in topic.argsort()[-10:]])
    print("\n")

topic_values = lda.transform(doc_term_matrix)
for i in range(n_topics):
    data[f'abstract_topic_{i}'] = topic_values[:, i]

# Assuming this part comes after you've fitted the LDA model
# lda = LatentDirichletAllocation(....)
# lda.fit(doc_term_matrix)

# PCA


# Apply PCA to the title TF-IDF features
pca_title = PCA(n_components=0.95)
title_tfidf_reduced = pca_title.fit_transform(title_tfidf_df)
title_tfidf_reduced_df = pd.DataFrame(title_tfidf_reduced)

# Apply PCA to the abstract TF-IDF features
pca_abstract = PCA(n_components=0.95)
abstract_tfidf_reduced = pca_abstract.fit_transform(abstract_tfidf_df)
abstract_tfidf_reduced_df = pd.DataFrame(abstract_tfidf_reduced)

# FIX PROBLEMS WITH FEATURE NAMES

# Add a prefix to the feature names for the title TF-IDF to make them unique
title_feature_names = [f"title_{name}" for name in vectorizer.get_feature_names_out()]
title_tfidf_df = pd.DataFrame(title_tfidf.toarray(), columns=title_feature_names)

# Add a prefix to the feature names for the abstract TF-IDF to make them unique
abstract_feature_names = [f"abstract_{name}" for name in abstract_vectorizer.get_feature_names_out()]
abstract_tfidf_df = pd.DataFrame(abstract_tfidf.toarray(), columns=abstract_feature_names)

# Now concatenate these dataframes with your main data
X = pd.concat([data.drop(['year', 'title', 'abstract', 'publisher', 'author', 'title_processed', 'abstract_processed'], axis=1),
               title_tfidf_df, abstract_tfidf_df], axis=1)

# Continue with your existing code for train-test split and model fitting


# RANDOM FOREST

data = data.copy()

X = pd.concat([data.drop(['year', 'title', 'abstract', 'publisher', 'author', 'title_processed', 'abstract_processed'], axis=1),
               title_tfidf_df, abstract_tfidf_df], axis=1).copy()

'''
# Combine reduced TF-IDF features with other features after PCA
X = pd.concat([data.drop(['title', 'abstract', 'publisher', 'author', 'title_processed', 'abstract_processed'] + list(title_tfidf_df.columns) + list(abstract_tfidf_df.columns), axis=1), 
               title_tfidf_reduced_df, abstract_tfidf_reduced_df], axis=1)
'''

y = data['year']

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initialize the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=0)

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

# MAE Before: 3.47
# MAE After Making Top Authors 100 : 3.58 (worse)
# MAE After limiting TF-IDF Features to 200 : 3.58 (worse)
# MAE After Advanced Text Analysis : 3.39

# FEATURE IMPORTANCE ANALYSIS


# Extracting feature importances
feature_importances = model.feature_importances_

# Matching feature names with their importances
feature_names = X_train.columns
importances = pd.Series(feature_importances, index=feature_names)

# Sorting the features by their importance
sorted_importances = importances.sort_values(ascending=False)

# Visualizing the top 20 most important features
plt.figure(figsize=(15, 3))
sorted_importances[:20].plot(kind='bar')
plt.title('Top 20 Feature Importances in Random Forest Model')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()

'''
# SELECTING FEATURES WITH high importances


importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Select a threshold to cut off features
threshold = 0.00001  # Example threshold
selected_features = X_train.columns
selected_features = X_train.columns[importances > threshold]

X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# Now retrain the model on this selected feature set
model.fit(X_train_selected, y_train)
y_pred = model.predict(X_test_selected)
new_mae = mean_absolute_error(y_test, y_pred)

print(f"New MAE with selected features: {new_mae}")

# Tried the importance of 0.1, 0.001, and even 0.00001, the MAE is not better than before

'''

# BLENDING

'''

# Train individual models (Random Forest can utilize multiple cores)
model1 = RandomForestRegressor(n_estimators=100, n_jobs=-1).fit(X_train, y_train)
model2 = SVR().fit(X_train, y_train)  # SVR does not support parallel processing

# Predict with individual models
pred1 = model1.predict(X_test)
pred2 = model2.predict(X_test)

# Blend predictions
blended_pred = (pred1 + pred2) / 2

# Evaluate blended predictions
blended_mae = mean_absolute_error(y_test, blended_pred)
print(f"MAE with Blended Model: {blended_mae}")

# MAE BLENDED: 3.545
'''

# STACKING MODELS

'''
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

# Define base models
estimators = [
    ('rf', RandomForestRegressor(n_estimators=100, n_jobs=-1)),
    ('svr', SVR()),
    ('dt', DecisionTreeRegressor())
]

# Create the stacking model
stack_model = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())

# Train the stacking model
stack_model.fit(X_train, y_train)

# Predict and evaluate
stack_pred = stack_model.predict(X_test)
stack_mae = mean_absolute_error(y_test, stack_pred)
print(f"MAE with Stacking Model: {stack_mae}")
'''