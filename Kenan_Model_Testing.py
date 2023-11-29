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
'''
# Lowercase Abstract
data['abstract_processed'] = data['abstract'].fillna('').str.lower()

# Feature Extraction: TF-IDF for 'abstract'
abstract_vectorizer = TfidfVectorizer(stop_words='english', max_features=500)  # Limit features to 500
abstract_tfidf = abstract_vectorizer.fit_transform(data['abstract_processed'])

# Convert 'abstract' TF-IDF to DataFrame
abstract_tfidf_df = pd.DataFrame(abstract_tfidf.toarray(), columns=abstract_vectorizer.get_feature_names_out())
'''
data.info()

# FROM FABIAN - Use Hashing Vectorizer instead of TDF IF
# In my case, the MAE increases from 3.35 to 3.36

from sklearn.feature_extraction.text import HashingVectorizer
# Lowercase Abstract
data['abstract_processed'] = data['abstract'].fillna('').str.lower()

# Feature Extraction: TF-IDF for 'abstract'
abstract_vectorizer = HashingVectorizer(n_features=1000, ngram_range=(1,1), lowercase=True)  # Limit features to 1000
abstract_tfidf = abstract_vectorizer.fit_transform(data['abstract_processed'])


# Convert 'abstract' TF-IDF to DataFrame
abstract_tfidf_df = pd.DataFrame(abstract_tfidf.toarray(), columns=[f'abstract{i}' for i in range(1000)])
# abstract_tfidf_df = pd.DataFrame(title_tfidf.toarray(), columns=vectorizer.get_feature_names_out())

i


# LENGTH OF ABSTRACT FEATURE

data['abstract_length'] = data['abstract_processed'].apply(len)

# FINDING WHETHER THE YEAR IS WRITTEN IN ABSTRACT / TITLE
 
import pandas as pd
import re

# Assume 'data' is your DataFrame and it has 'title' and 'abstract' columns

# Function to check for a year between 1950 and 2022 in the text
def year_in_text(text):
    # Convert None to an empty string
    if text is None:
        return 0
    # This regex will match any four consecutive digits between 1950 and 2022
    year_pattern = r'\b(19[5-9]\d|20[0-1]\d|2022)\b'
    return int(bool(re.search(year_pattern, text)))

# Apply the function to the 'abstract' column
data['year_in_abstract'] = data['abstract'].apply(year_in_text)

# Apply the function to title and abstract columns
data['year_in_title'] = data['title'].apply(year_in_text)

data = data.copy()
# Display the new columns
print(data[['year_in_title', 'year_in_abstract']])

data['year_in_title'].describe

# Assuming 'year' is your target variable and is already defined
year_in_title_corr = data['year'].corr(data['year_in_title'])
year_in_abstract_corr = data['year'].corr(data['year_in_abstract'])

print(f"Correlation between 'year' and 'year_in_title': {year_in_title_corr}")
print(f"Correlation between 'year' and 'year_in_abstract': {year_in_abstract_corr}")


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

# ... [your existing code] ...

# Start the prediction timer
predict_start_time = time.time()

# Predict on the testing set
y_pred_uncapped = model.predict(X_test)

# Cap the predictions to be within the range 1970 to 2021
y_pred = np.clip(y_pred_uncapped, 1970, 2021)

# Stop the prediction timer and print the time taken
predict_end_time = time.time()
print(f"Prediction Time: {predict_end_time - predict_start_time} seconds")

# Calculate Mean Absolute Error using the capped predictions
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error with capped predictions: {mae}")

# Start the prediction timer
predict_start_time = time.time()

# Predict on the testing set
y_pred = model.predict(X_test)

# Stop the prediction timer and print the time taken
predict_end_time = time.time()
print(f"Prediction Time: {predict_end_time - predict_start_time} seconds")

# 28 Nov Addition: cap results to be 1970-2021

def cap_predictions(predictions, lower_bound, upper_bound):
    # Cap predictions to be within lower_bound and upper_bound
    predictions_capped = np.clip(predictions, lower_bound, upper_bound)
    return predictions_capped

# Assume model is your trained RandomForestRegressor and X_test is your test set
predictions = model.predict(X_test)

# Cap the predictions to be within the range 1970 to 2021
predictions_capped = cap_predictions(predictions, 1980, 2021)

# Now use predictions_capped for further evaluation
mae_capped = mean_absolute_error(y_test, predictions_capped)
print(f"Mean Absolute Error with capped predictions: {mae_capped}")


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


# RFE
# Recursive Feature Elimination (RFE) is a feature selection method that fits 
# a model and removes the weakest feature (or features) until the specified 
# number of features is reached. It's a way of selecting important features 
# by recursively considering smaller and smaller sets of features.

'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_absolute_error

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initialize the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=0)

# Initialize RFE with the random forest model
selector = RFE(rf_model, n_features_to_select=500, step=50)  # n_features_to_select should be set based on domain knowledge or experimentation
selector = selector.fit(X_train, y_train)

# Transform training and test sets
X_train_rfe = selector.transform(X_train)
X_test_rfe = selector.transform(X_test)

# Fit the model on the reduced dataset
rf_model.fit(X_train_rfe, y_train)

# Predict and evaluate
y_pred = rf_model.predict(X_test_rfe)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error with RFE: {mae}")

# Took 2 hours, MAE 3.48
'''

# Now the opposite way: 
# adding features one by one, up to 20 features, and keep track of the best MAE

'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit a model on all features to get the feature importances
model_all_features = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
model_all_features.fit(X_train, y_train)
feature_importances = model_all_features.feature_importances_

# Sort features by their importance
sorted_features = [feature for _, feature in sorted(zip(feature_importances, X_train.columns), reverse=True)]

# Incrementally add features and track MAE, up to 50 features
max_features = 100
mae_scores = []
features_used = []

for i in range(1, max_features + 1):
    selected_features = sorted_features[:i]
    features_used.append(selected_features)
    
    # Train model with the selected features
    model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)
    model.fit(X_train[selected_features], y_train)
    
    # Make predictions and calculate MAE
    y_pred = model.predict(X_test[selected_features])
    mae = mean_absolute_error(y_test, y_pred)
    mae_scores.append(mae)

    print(f"MAE with top {i} features: {mae}")

# Find the number of features with the lowest MAE
optimal_feature_count = mae_scores.index(min(mae_scores)) + 1
optimal_features = features_used[optimal_feature_count - 1]

print(f"Optimal number of features: {optimal_feature_count}")
print(f"Features for optimal MAE: {optimal_features}")

# Top 20 features: MAE 4.3, top 50: MAE of 4.20 top 100: 3.91
'''

# ERROR ANALYSIS


import pandas as pd

# Assuming y_test and y_pred are defined (from your test set and model predictions)

# Create a DataFrame for analysis
error_analysis_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
error_analysis_df['Error'] = error_analysis_df['Actual'] - error_analysis_df['Predicted']

# Analyze error distribution
error_analysis_df['Error'].describe()

# You might want to visualize or further analyze where the largest errors are occurring

import matplotlib.pyplot as plt
import seaborn as sns

# Histogram of errors
plt.figure(figsize=(10, 6))
sns.histplot(error_analysis_df['Error'], bins=50, kde=True)
plt.title('Distribution of Prediction Errors')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.show()

# Boxplot of errors
plt.figure(figsize=(10, 6))
sns.boxplot(x=error_analysis_df['Error'])
plt.title('Box Plot of Prediction Errors')
plt.xlabel('Prediction Error')
plt.show()

# The problem is that there are a lot of extreme outliers


# HANDLING OUTLIERS

'''
# Calculate IQR
Q1 = error_analysis_df['Error'].quantile(0.25)
Q3 = error_analysis_df['Error'].quantile(0.75)
IQR = Q3 - Q1

# Define outliers
outlier_threshold_upper = Q3 + 1.5 * IQR
outlier_threshold_lower = Q1 - 1.5 * IQR

# Filter outliers
outliers = error_analysis_df[(error_analysis_df['Error'] > outlier_threshold_upper) | 
                             (error_analysis_df['Error'] < outlier_threshold_lower)]

outliers

# Merge outliers with original data to analyze them
outliers_full_data = outliers.join(X_test, how='left')


# IF WE REMOVE THE OUTLIERS
# Assuming 'outliers' is a DataFrame containing the outliers with their indices
outlier_indices = outliers.index.tolist()

# Ensure that the indices are in the X_train index
outlier_indices = [index for index in outlier_indices if index in X_train.index]

# Remove these indices from the training set
X_train_filtered = X_train.drop(index=outlier_indices)
y_train_filtered = y_train.drop(index=outlier_indices)

# Retrain the model with the filtered dataset
model.fit(X_train_filtered, y_train_filtered)

# Make predictions on the test set and calculate MAE
y_pred = model.predict(X_test)
mae_filtered = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error after removing outliers: {mae_filtered}")

# MAE 3.43

# IF WE TRANSFORM THE OUTLIERS

# Apply a logarithmic transformation to 'year'
y_train_log_transformed = np.log(y_train - y_train.min() + 1)

# Retrain the model
model.fit(X_train, y_train_log_transformed)

# Make predictions and reverse the transformation
y_pred_log_transformed = model.predict(X_test)
y_pred_transformed_back = np.exp(y_pred_log_transformed) + y_train.min() - 1
mae_transformed = mean_absolute_error(y_test, y_pred_transformed_back)
print(f"Mean Absolute Error after log transformation: {mae_transformed}")

# I WAS SO HOPEFUL, but then the MAE is 3.50..

# IF WE CAP OUTLIERS

# Define capping function
def cap_series(s, lower_threshold, upper_threshold):
    return s.apply(lambda x: min(max(x, lower_threshold), upper_threshold))

# Cap the 'year' in the training set
y_train_capped = cap_series(y_train, outlier_threshold_lower, outlier_threshold_upper)

# Retrain the model
model.fit(X_train, y_train_capped)

# Make predictions as before
y_pred_capped = model.predict(X_test)
mae_capped = mean_absolute_error(y_test, y_pred_capped)
print(f"Mean Absolute Error after capping outliers: {mae_capped}")
'''

# CONTINUE ERROR ANALYSIS

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

# Assuming y_test and y_pred are already defined
error_analysis_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
error_analysis_df['Error'] = error_analysis_df['Actual'] - error_analysis_df['Predicted']

# Define a high-error threshold, here I'm using the 75th percentile of absolute errors
high_error_threshold = error_analysis_df['Error'].abs().quantile(0.75)

# Identify high-error instances
high_error_instances = error_analysis_df[abs(error_analysis_df['Error']) > high_error_threshold]

# Combine the high-error instances with the original data to get feature values
high_error_full_data = high_error_instances.join(X_test, how='left')

# Conduct analysis on the high-error instances
# Statistical summary of high-error instances
high_error_summary = high_error_full_data.describe()

# Compare with low-error instances
low_error_instances = error_analysis_df[abs(error_analysis_df['Error']) <= high_error_threshold]
low_error_full_data = low_error_instances.join(X_test, how='left')
low_error_summary = low_error_full_data.describe()

# Correlation of features with errors
feature_error_correlation = high_error_full_data.corrwith(error_analysis_df['Error'])

'''
# Visualize high-error feature distributions compared to low-error instances
for feature in X_test.columns:
    plt.figure(figsize=(10, 6))
    sns.kdeplot(high_error_full_data[feature], label='High Error')
    sns.kdeplot(low_error_full_data[feature], label='Low Error')
    plt.title(f'Distribution of {feature} for High vs. Low Error Instances')
    plt.legend()
    plt.show()
'''
# Print statistical summaries
print("High error instances summary:")
print(high_error_summary)
print("\nLow error instances summary:")
print(low_error_summary)

# Print correlation of features with errors
print("\nCorrelation of features with errors:")
print(feature_error_correlation.sort_values(ascending=False))

# FEATURE RELEVANCE

# Finding the feature which highly correlates with the error
# Assuming feature_error_correlation is already defined and sorted by correlation with error
high_corr_features = feature_error_correlation.abs().sort_values(ascending=False).head(5).index.tolist()

# Display the top correlated features and their correlation values
print(feature_error_correlation[high_corr_features])

# Let's assume `high_corr_features` is the list of features sorted by their correlation with the error
# And the first element is 'Actual', which we want to skip

# Exclude the 'Actual' key from your feature list, assuming 'Actual' is the first in the list
for feature in high_corr_features[1:]:  # This excludes the first element, 'Actual'
    # Check if the feature exists in X_train to avoid KeyError
    if feature in X_train.columns:
        sns.jointplot(x=X_train[feature], y=y_train, kind='reg')
        plt.xlabel(feature)
        plt.ylabel('Year')
        plt.title(f'Relationship of {feature} with Year')
        plt.show()


# Drop 3 features
X_train_refined = X_train.drop('title_russian','publisher_Association for Computational Linguistics','Relationship of publisher_Association for Computational Linguistics with Year', axis=1)
X_test_refined = X_test.drop('title_russian','publisher_Association for Computational Linguistics','Relationship of publisher_Association for Computational Linguistics with Year', axis=1)

# Re-train the model
model.fit(X_train_refined, y_train)

# Evaluate the model
y_pred_refined = model.predict(X_test_refined)
mae_refined = mean_absolute_error(y_test, y_pred_refined)
print(f"Refined MAE: {mae_refined}")

#MAE Still 3.39

# Correlation with errors, but for non-binary features
# (the ones that are not one hot encoded)

# Assuming X_train and y_train are already defined and are your feature set and target variable, respectively.

# Identify non-binary features
non_binary_features = [col for col in X_train.columns if len(X_train[col].unique()) > 2]

# Now, let's perform error correlation on non-binary features only
# First, let's calculate the errors
y_pred = model.predict(X_train)
errors = y_train - y_pred

# Create a DataFrame from non-binary features
X_train_non_binary = X_train[non_binary_features]

# Calculate the correlation of each non-binary feature with the error
feature_error_correlation_non_binary = X_train_non_binary.apply(lambda x: x.corr(errors))

# Sort the features by their correlation with error
sorted_correlation_non_binary = feature_error_correlation_non_binary.abs().sort_values(ascending=False)

# Now you can print or visualize the sorted correlations
print(sorted_correlation_non_binary)
