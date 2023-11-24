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

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

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

from sklearn.decomposition import PCA
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error

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

# NEURAL NETWORKS

# Import TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Convert X_train and X_test to float
X_train = X_train.astype(float)
X_test = X_test.astype(float)

# Define and compile the neural network model
nn_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.1),
    Dense(64, activation='relu'),
    Dense(1)  # Output layer
])
nn_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
nn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Predict and evaluate
nn_pred = nn_model.predict(X_test).flatten()
nn_mae = mean_absolute_error(y_test, nn_pred)
print(f"MAE with Neural Network: {nn_mae}")

# MAE 12.9 .....

nn_model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.1),
    Dense(64, activation='relu'),
    Dense(1)  # Output layer
])

nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                 loss='mean_squared_error')

nn_model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.2)

nn_pred = nn_model.predict(X_test).flatten()
nn_mae = mean_absolute_error(y_test, nn_pred)
print(f"MAE with Neural Network: {nn_mae}")

# Now the MAE is 19..
'''


# I asked chat GPT to optimize the dataset for neural network and run it again
# while I sleep. Here goes..

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.optimizers import Adam

# Feature selection using PCA
pca = PCA(n_components=0.95)  # Adjust n_components as needed
X_pca = pca.fit_transform(X)

# Data normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pca)

# Neural network architecture
def create_model(learning_rate=0.01):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_scaled.shape[1],)),
        Dropout(0.1),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# Wrapping Keras model with KerasRegressor for compatibility with scikit-learn
nn_model = KerasRegressor(build_fn=create_model, epochs=10, batch_size=32, verbose=0)

# Hyperparameters to tune
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'batch_size': [32, 64, 128]
}

# Grid search
grid = GridSearchCV(estimator=nn_model, param_grid=param_grid, cv=3)
grid_result = grid.fit(X_scaled, y)

# Best parameters
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# Train the best model
best_model = grid_result.best_estimator_
history = best_model.fit(X_scaled, y, validation_split=0.2)

# Predict and evaluate
y_pred = best_model.predict(scaler.transform(pca.transform(X_test)))
nn_mae = mean_absolute_error(y_test, y_pred)
print(f"MAE with Optimized Neural Network: {nn_mae}")

# MAE 3.39! It has reached parity with Random Forest

# Next Try
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.callbacks import EarlyStopping

# Define the model creation function
def create_optimized_model(learning_rate=0.01):
    model = Sequential([
        Dense(256, activation='relu', input_shape=(X_scaled.shape[1],)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# Initialize KerasRegressor with the model creation function
nn_optimized_model = KerasRegressor(model=create_optimized_model, epochs=100, batch_size=32, verbose=0)

# Define the parameter grid for grid search
param_grid_optimized = {
    'model__learning_rate': [0.001, 0.005, 0.01],
    'model__batch_size': [32, 64]
}

# Perform grid search
grid_optimized = GridSearchCV(estimator=nn_optimized_model, param_grid=param_grid_optimized, cv=3, n_jobs=-1)
grid_result_optimized = grid_optimized.fit(X_scaled, y)

# Output best parameters and score
print("Best: %f using %s" % (grid_result_optimized.best_score_, grid_result_optimized.best_params_))

# Get the best model from grid search
best_optimized_model = grid_result_optimized.best_estimator_.model

# Train the best model
history_optimized = best_optimized_model.fit(X_scaled, y, epochs=100, batch_size=32, validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', patience=10)])

# Predict with the optimized model
y_pred_optimized = best_optimized_model.predict(scaler.transform(pca.transform(X_test))).flatten()
nn_mae_optimized = mean_absolute_error(y_test, y_pred_optimized)

# Output MAE
print(f"MAE with Further Optimized Neural Network: {nn_mae_optimized}")
