"""
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

# WHOLE DATA

data = pd.read_json('train.json')
print(data.head())

# Basic info about data
print(data.info())

# Summary statistics for numerical fields
print(data.describe())


# YEAR COLUMN


# Distribution of the 'year' field
print(data['year'].value_counts())

# Set the aesthetic style of the plots
sns.set()

# Plotting the distribution of the 'year' column
plt.figure(figsize=(10, 6))
sns.histplot(data['year'], kde=False, color='blue', bins=30)
plt.title('Distribution of Publication Years')
plt.xlabel('Year')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


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

# Counting How Many item in each entrytype (used sum because dummy coded uses boolean
# this code counted how many 'true' for each column)
entrytype_counts = data[['entrytype_article', 'entrytype_inproceedings', 'entrytype_proceedings']].sum()
print(entrytype_counts)


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

# Box Plot
# Filter out to include only the top N publishers for a clearer plot
top_publishers = publisher_counts.index[:30]  # Top 30 publishers
filtered_data = data[data['publisher'].isin(top_publishers)]
filtered_data

# Create a box plot
plt.figure(figsize=(15, 8))
sns.boxplot(x='publisher', y='year', data=filtered_data)
plt.xticks(rotation=45)
plt.title('Distribution of Publication Years for Top Publishers')
plt.show()


# Bar Plot
mean_years.plot(kind='bar', figsize=(15, 8))
plt.title('Mean Publication Year for Each Publisher')
plt.xlabel('Publisher')
plt.ylabel('Mean Publication Year')
plt.xticks(rotation=90)
plt.ylim(1940, mean_years.max() + 1)  # Set y-axis limits
plt.show()

# Since there are 8200 NA from 65000 items, the decision is not as easy
# There are 3 options: impute the missing data, delete the 8200 instances, or mark all 8200 of them as 'unknown'


# AUTHOR COLUMN

# Flatten the list of authors
all_authors = set()
for authors_list in data['author'].dropna():
    all_authors.update(authors_list)

len(all_authors)
# Now all_authors contains all unique authors

from collections import Counter

# Initialize a Counter object to hold author frequencies
author_frequencies = Counter()

# Iterate through the author lists and update the counter
for authors_list in data['author'].dropna():
    author_frequencies.update(authors_list)

# Now author_frequencies contains the count of each author
author_frequencies

# Determine the number of top authors you want to consider, e.g., top 100
num_top_authors = 100

# Get the most common authors
most_common_authors = author_frequencies.most_common(num_top_authors)

# Print the most common authors
for author, count in most_common_authors:
    print(f"{author}: {count}")

most_common_authors

# Assuming 'most_common_authors' contains your top authors
top_authors = [author for author, count in most_common_authors]

# Dictionary to hold year distribution data for each top author
year_distributions = {}

for author in top_authors:
    # Filter data for the current author
    author_data = data[data['author'].apply(lambda x: author in x if isinstance(x, list) else False)]
    
    # Get year distribution for this author
    year_distributions[author] = author_data['year'].describe()

year_distributions

# Print the year distribution for each top author
for author, distribution in year_distributions.items():
    print(f"Year distribution for {author}:")
    print(distribution, "\n")


    
# Categorizing authors based on publication count
frequency_categories = {'1-5 publications': 0, '6-20 publications': 0, '21-50 publications': 0, '50-100 publications': 0, '100+ publications': 0}
for count in author_frequencies.values():
    if 1 <= count <= 5:
        frequency_categories['1-5 publications'] += 1
    elif 6 <= count <= 20:
        frequency_categories['6-20 publications'] += 1
    elif 21 <= count <= 50:
        frequency_categories['21-50 publications'] += 1
    elif 51 <= count <= 100:
        frequency_categories['50-100 publications'] += 1
    else:
        frequency_categories['100+ publications'] +=1

frequency_categories

# Categories and counts for plotting
categories, counts = zip(*frequency_categories.items())

# Create a bar plot
plt.figure(figsize=(10, 6))
plt.bar(categories, counts, color='skyblue')
plt.xlabel('Number of Publications')
plt.ylabel('Number of Authors')
plt.title('Distribution of Authors by Number of Publications')
plt.show()

# So, most authors have few publications. For example, 
# 59000 authors have 1-5 publications only.
# Which means the majority does not have predictive power?
# An idea: use one-hot encoding, but only for authors with 100+ publications (67 people, so 67 added features)
# or 50+ publications, so 253 + 67 = 320 people (so 320 features)

# Idea from chat GPT, instead of author names, maybe the number of authors in a publication can be a predictor
# Let's make that a new feature

# Assuming each entry in the 'author' column is a list of authors
data['author_count'] = data['author'].apply(lambda x: len(x) if isinstance(x, list) else 0)

# Identify authors with 50+ publications
prolific_authors = [author for author, count in author_frequencies.items() if count >= 50]

# One-hot encode these authors
for author in prolific_authors:
    data[f'author_{author}'] = data['author'].apply(lambda x: author in x if isinstance(x, list) else False)

prolific_authors

# Now our dataframe has 347 columns..
data.shape

# CHECKING MODEL SO FAR

# I'll do a simple random forest regression check on the current data, without
# including the title and abstract column for now

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd
import time

# Sample data loading (replace this with your actual dataset)
# data = pd.read_csv('your_dataset.csv')

# Exclude title, abstract, and publisher column. Not 
# X = data.drop(['year', 'title', 'abstract', 'publisher', 'author'], axis=1)
# y = data['year']

# Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
# model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)

# Start the timer for training
# train_start_time = time.time()

# Train the model
# model.fit(X_train, y_train)

# Stop the timer for training and print the time taken
# train_end_time = time.time()
# print(f"Training Time: {train_end_time - train_start_time} seconds")

# Start the timer for prediction
# predict_start_time = time.time()

# Predict on the testing set
# y_pred = model.predict(X_test)

# Stop the timer for prediction and print the time taken
# predict_end_time = time.time()
# print(f"Prediction Time: {predict_end_time - predict_start_time} seconds")

# Calculate Mean Absolute Error
# mae = mean_absolute_error(y_test, y_pred)
# print(f"Mean Absolute Error: {mae}")

# The MAE is 6.41 and the time taken is 2 minutes
# Damn it's even worse than the baseline


# TITLE and ABSTRACT COLUMNS

# Make Lower case
data['title_processed'] = data['title'].str.lower()

# Feature Extraction: TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=500)  # Limit features to 500 for simplicity
title_tfidf = vectorizer.fit_transform(data['title_processed'])

# Convert to DataFrame
title_tfidf_df = pd.DataFrame(title_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
title_tfidf_df

# Combining TF-IDF features with other features (excluding raw text columns)
# X = pd.concat([data.drop(['year', 'title', 'abstract', 'publisher', 'author', 'title_processed'], axis=1), title_tfidf_df], axis=1)
# y = data['year']

# Splitting the dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Now you can proceed to train a model with X_train and y_train
# model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)

# Start the timer for training
# train_start_time = time.time()

# Train the model
# model.fit(X_train, y_train)
 
# Stop the timer for training and print the time taken
# train_end_time = time.time()
# print(f"Training Time: {train_end_time - train_start_time} seconds")

# Start the timer for prediction
# predict_start_time = time.time()

# Predict on the testing set
# y_pred = model.predict(X_test)

# Stop the timer for prediction and print the time taken
# predict_end_time = time.time()
# print(f"Prediction Time: {predict_end_time - predict_start_time} seconds")

# Calculate Mean Absolute Error
# mae = mean_absolute_error(y_test, y_pred)
# print(f"Mean Absolute Error: {mae}")

# With tfidf Vectorizer on the Title column, the MAE becomes 5.34 and the
# Training time is 6 minutes
# Let me try to do it to the abstract column too

# Lowercase Abstract
data['abstract_processed'] = data['abstract'].fillna('').str.lower()

# Feature Extraction: TF-IDF for 'abstract'
abstract_vectorizer = TfidfVectorizer(stop_words='english', max_features=500)  # Limit features to 500
abstract_tfidf = abstract_vectorizer.fit_transform(data['abstract_processed'])

# Convert 'abstract' TF-IDF to DataFrame
abstract_tfidf_df = pd.DataFrame(abstract_tfidf.toarray(), columns=abstract_vectorizer.get_feature_names_out())

# Combining TF-IDF features with other features
# X = pd.concat([data.drop(['year', 'title', 'abstract', 'publisher', 'author', 'title_processed', 'abstract_processed'], axis=1),
#               title_tfidf_df, abstract_tfidf_df], axis=1).copy()
# y = data['year']

# Splitting the dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
# model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)

# Start the training timer
# train_start_time = time.time()

# Train the model
# model.fit(X_train, y_train)

# Stop the training timer and print the time taken
# train_end_time = time.time()
# print(f"Training Time: {train_end_time - train_start_time} seconds")

# Start the prediction timer
# predict_start_time = time.time()

# Predict on the testing set
# y_pred = model.predict(X_test)

# Stop the prediction timer and print the time taken
# predict_end_time = time.time()
# print(f"Prediction Time: {predict_end_time - predict_start_time} seconds")

# Calculate Mean Absolute Error
# mae = mean_absolute_error(y_test, y_pred)
# print(f"Mean Absolute Error: {mae}")

# With tfidf Vectorizer on the Title and Abstract Column, the MAE goes even further down, to 4.40

# Now, back to publisher, the only column that isn't solved yet:

# PUBLISHER COLUMN PART 2
# Part 1: one-hot encode, and then keep the missing value as 'unknown'
# Part 2: remove the missing value
# compare MAE in part 1 and 2

data['publisher'].fillna('Unknown', inplace=True)

# One-hot encoding of the 'publisher' column
publisher_dummies = pd.get_dummies(data['publisher'], prefix='publisher')

# Join the one-hot encoded columns back to the original DataFrame
data = pd.concat([data, publisher_dummies], axis=1)


# NOW, LET'S DO RANDOM FOREST AGAIN WITH ALL THE COLUMNS TREATED


X = pd.concat([data.drop(['year', 'title', 'abstract', 'publisher', 'author', 'title_processed', 'abstract_processed'], axis=1),
               title_tfidf_df, abstract_tfidf_df], axis=1).copy()
y = data['year']

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)

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
# MAE: 3.53


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

# Plotting Author count by year
plt.figure(figsize=(15, 8))
sns.boxplot(data=data, x='year', y='author_count')
plt.xticks(rotation=90)  # Rotate x labels for better readability if needed
plt.title('Distribution of Author Count Over Years')
plt.xlabel('Year')
plt.ylabel('Author Count')
plt.show()

# Let's try again doing the Random Forest without the 'publisher_unknown' feature

# X = pd.concat([data.drop(['year', 'title', 'abstract', 'publisher','publisher_Unknown', 'author', 'title_processed', 'abstract_processed'], axis=1),
#               title_tfidf_df, abstract_tfidf_df], axis=1).copy()
# y = data['year']

# Splitting the dataset
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
# model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)

# Start the training timer
# train_start_time = time.time()

# Train the model
# model.fit(X_train, y_train)

# Stop the training timer and print the time taken
# train_end_time = time.time()
# print(f"Training Time: {train_end_time - train_start_time} seconds")

# Start the prediction timer
# predict_start_time = time.time()

# Predict on the testing set
# y_pred = model.predict(X_test)

# Stop the prediction timer and print the time taken
# predict_end_time = time.time()
# print(f"Prediction Time: {predict_end_time - predict_start_time} seconds")

# Calculate Mean Absolute Error
# mae = mean_absolute_error(y_test, y_pred)
# print(f"Mean Absolute Error: {mae}")

# Without the publisher_unknown, the MAE Increased to 3.67


# HYPERPARAMETER TUNING

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

# Define the parameter grid to search
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [10, 20, 30, 40, 50, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialize the Random Forest Regressor
rf = RandomForestRegressor(random_state=42)

# Initialize the RandomizedSearchCV object
rf_random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=100, cv=5, verbose=2, random_state=42, n_jobs=-1)

# Fit the random search model
rf_random_search.fit(X_train, y_train)

# Print the best parameters found by RandomizedSearchCV
print("Best parameters found: ", rf_random_search.best_params_)

# Evaluate the best model found on the test set
best_model = rf_random_search.best_estimator_
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error with best model: {mae}")

# In hyperparameter tuning, the best model is
# 'n_estimators': 200, 'min_samples_split': 5, 'min_samples_leaf': 1,
# 'max_features': 'sqrt', 'max_depth': None, 'bootstrap': True
# With MAE of 3.59
# The MAE was worse than before, now I'm confused lol

# This one below is still the best one, with the MAE of 3.53. 
# But if can't be replicated, might indicate overfitting

X = pd.concat([data.drop(['year', 'title', 'abstract', 'publisher', 'author', 'title_processed', 'abstract_processed'], axis=1),
               title_tfidf_df, abstract_tfidf_df], axis=1).copy()
y = data['year']

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)

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
# MAE: 3.53

# CROSS VALIDATION (5-fold)

# Initialize the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=42)

# Define the number of folds for cross-validation
num_folds = 5

# Perform cross-validation and then calculate the mean absolute error for each fold
# Note: cross_val_score by default returns the negative mean absolute error in regression tasks
# to maximize the score, so we need to take the absolute value of its output
mae_scores = -cross_val_score(model, X, y, cv=num_folds, scoring='neg_mean_absolute_error')

# Calculate the average and standard deviation of the cross-validation MAE scores
average_mae = mae_scores.mean()
std_dev_mae = mae_scores.std()

print(f"Average MAE from cross-validation: {average_mae}")
print(f"Standard Deviation of MAE from cross-validation: {std_dev_mae}")

# Average MAE from Cross-Validation = 3.44, STDev 0.05
# Cross Validation Seems to Indicate no overfitting. If anything, in some split, 
# the MAE can go as low as 3.39


# TRYING OTHER REGRESSION METHODS

# Linear Regression

from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred = linear_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Linear Regression MAE: {mae}")
# MAE: 4.17

# Ridge Regression

from sklearn.linear_model import Ridge

ridge_model = Ridge(alpha=1.0)  # alpha is the regularization strength
ridge_model.fit(X_train, y_train)
y_pred = ridge_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Ridge Regression MAE: {mae}")
# MAE: 4.17

# Lasso Regression

from sklearn.linear_model import Lasso

lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)
y_pred = lasso_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Lasso Regression MAE: {mae}")
# MAE: 5.89

# Support Vector Regression
from sklearn.svm import SVR

svr_model = SVR(C=1.0, epsilon=0.2)  # C is the regularization parameter
svr_model.fit(X_train, y_train)
y_pred = svr_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"SVR MAE: {mae}")

# Took so long (almost 1h), MAE: 3.94

# Gradient Boosting Regression
from sklearn.ensemble import GradientBoostingRegressor

gbr_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
gbr_model.fit(X_train, y_train)
y_pred = gbr_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Gradient Boosting MAE: {mae}")

# MAE: 4.42

# None of the other regression methods get a better score than random forest.
