"""
Here are how each feature is treated in this file:
1. Entrytype: One-hot encoded as 3 variables
2. Editor: too many Missing Values, so it's dropped
3. Publisher: One-hot encoded as 120 unique variables. the Missing Values are renamed: 'unknown_publisher'
4. Author: One-hot encoded the authors with 50+ publications. Also, added another feature: Author_count, which is the number of author in each paper
5. Title: Tf Idf Vectorizer, top 500 words
6. Abstract: Tf Idf Vectorizer, top 500 words

Then, from those cleaned data, this is what has been done:
1. Random Forest Regressor, got MAE of 3.53
2. Feature Importance Analysis
3. Cross-Validation (5-fold)
4. Hyperparameter Tuning
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import KBinsDiscretizer
from transformers import DistilBertTokenizer, DistilBertModel
import torch

data = pd.read_json('input/train.json')


# -------------------------------------------------Year column-------------------------------------------------#


# -------------------------------------------------Entrytype column-------------------------------------------------#

# One-hot encode the 'ENTRYTYPE' column
entrytype_dummies = pd.get_dummies(data['ENTRYTYPE'], prefix='entrytype')

# Concatenate the new columns with the original dataframe
data = pd.concat([data, entrytype_dummies], axis=1)

# Drop the original 'ENTRYTYPE' column
data.drop('ENTRYTYPE', axis=1, inplace=True)

# Counting How Many item in each entrytype (used sum because dummy coded uses boolean
# this code counted how many 'true' for each column)
entrytype_counts = data[['entrytype_article', 'entrytype_inproceedings', 'entrytype_proceedings']].sum()

# -------------------------------------------------Editor column-------------------------------------------------#

# Count null values in the 'editor' column
null_count = data['editor'].isnull().sum()

data.drop('editor', axis=1, inplace=True)


# -------------------------------------------------Publisher column-------------------------------------------------#
# Display unique values in the 'publisher' column
unique_publishers = data['publisher'].unique()

# Count of unique values
# There are 120 Unique Publishers

# Display the frequency of each publisher
publisher_counts = data['publisher'].value_counts()

# Count null values in the 'publisher' column
null_count = data['publisher'].isnull().sum()
# 8201 null values from 65914 Data Points

# Relationship Between Publisher and Year
# Calculate mean year for each publisher
mean_years = data.groupby('publisher')['year'].mean().sort_values()

# Box Plot
# Filter out to include only the top N publishers for a clearer plot
top_publishers = publisher_counts.index[:30]  # Top 30 publishers
# 30 publishers = MAE: 4.548
# 50 publishers = MAE: 4.548
filtered_data = data[data['publisher'].isin(top_publishers)]

# Since there are 8200 NA from 65000 items, the decision is not as easy
# What is done below: impute the NA as 'unknown_publisher'
data['publisher'].fillna('Unknown', inplace=True)

# One-hot encoding of the 'publisher' column --> 4.5
publisher_dummies = pd.get_dummies(data['publisher'], prefix='publisher')

# Join the one-hot encoded columns back to the original DataFrame
data = pd.concat([data, publisher_dummies], axis=1)
# ---------------------------------------------------------------------

# -------------------------------------------------Author column-------------------------------------------------#

# -------------------------------------------------Title and abstract-------------------------------------------------#

# Make Lower case
data['title_processed'] = data['title'].str.lower()

# Feature Extraction: TF-IDF
# vectorizer = HashingVectorizer(n_features=1000)  # Limit features to 1000 for simplicity
vectorizer = CountVectorizer(analyzer='word', ngram_range=(1,1), lowercase=True) 
# with stripaccent --> 4.54
title_tfidf = vectorizer.fit_transform(data['title_processed'])

# Convert to DataFrame
title_tfidf_df = pd.DataFrame(title_tfidf.toarray(), columns=vectorizer.get_feature_names_out())

# Lowercase Abstract
data['abstract_processed'] = data['abstract'].fillna('').str.lower()

# data['abstract_length'] = data['abstract_processed'].apply(len)

# Feature Extraction: TF-IDF for 'abstract'
abstract_vectorizer = HashingVectorizer(n_features=1000, ngram_range=(1,1), lowercase=True)  # Limit features to 1000
abstract_tfidf = abstract_vectorizer.fit_transform(data['abstract_processed'])

# Convert 'abstract' TF-IDF to DataFrame
abstract_tfidf_df = pd.DataFrame(abstract_tfidf.toarray(), columns=[f'abstract{i}' for i in range(1000)])
# abstract_tfidf_df = pd.DataFrame(title_tfidf.toarray(), columns=vectorizer.get_feature_names_out())



# -------------------------------------------------Random Forest-------------------------------------------------#

# NOW, LET'S DO RANDOM FOREST WITH ALL THE FEATURES TREATED


#import xgboost as xgb
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# import lightgbm as lgb
# kb = KBinsDiscretizer(n_bins=20, strategy='uniform', encode='onehot-dense', subsample = None)

X = pd.concat([data.drop(['year', 'title', 'abstract', 'publisher', 'author', 'title_processed', 'abstract_processed'], axis=1),
                title_tfidf_df, abstract_tfidf_df], axis=1).copy()
# trim_percentage = 5
# lower_bound = np.percentile(data['year'], trim_percentage)
# mask = data['year'] >= lower_bound
# y = data.loc[mask, 'year']
y = data['year']
y = np.log1p(y)
# kb.fit(X)
# X_binned = kb.transform(X)

weights = y / y.max() 

# Splitting the dataset
X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(X, y, weights, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
# model = RandomForestRegressor(n_estimators=250, n_jobs=-1, random_state=42)
model = RandomForestRegressor(n_estimators=150, max_depth=3, n_jobs=-1, random_state=42)

# model = GradientBoostingRegressor(learning_rate=0.1, n_estimators=250, max_depth=15, random_state=42)

# Start the training timer
train_start_time = time.time()

# Train the model
model.fit(X_train, y_train, sample_weight=weights_train)

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

# Transform the predicted values back to the original scale
y_pred_original_scale = np.expm1(y_pred)
y_test_original_scale = np.expm1(y_test)

# Calculate Mean Absolute Error
mae = mean_absolute_error(y_test_original_scale, y_pred_original_scale)
print(f"Mean Absolute Error: {mae}")
