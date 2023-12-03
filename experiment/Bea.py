


"""changes:
ABSTRACT: fill missing values with no_abstract + use countVectonizer
 + also variable len of abstarct + max_features to 2000
TITLE: max features to 2000 + use_idf=False
EDITOR: adding editor column with just number of editors

--> error in total decreased from 3.53 to 3.274"""

# max_features to 1000 = 3.321
# max_features to 2000 = 3.277



from googletrans import Translator, constants
from pprint import pprint

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

# WHOLE DATA

data = pd.read_json('input/train.json')
data.columns


# ENTRYTYPE COLUMN

# One-hot encode the 'ENTRYTYPE' column
entrytype_dummies = pd.get_dummies(data['ENTRYTYPE'], prefix='entrytype')

# Concatenate the new columns with the original dataframe
data = pd.concat([data, entrytype_dummies], axis=1)


# EDITOR COLUMN

# data.drop('editor', axis=1, inplace=True)
""" adding an editor column that just shows number of editors decreases error by 0.09"""
data['editor_count'] = data['editor'].apply(lambda x: len(x) if isinstance(x, list) else 0)



# PUBLISHER COLUMN

# What is done below: impute the NA as 'unknown_publisher'
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

len(all_authors)
# Now all_authors contains all unique authors

from collections import Counter

# Initialize a Counter object to hold author frequencies
author_frequencies = Counter()

# Iterate through the author lists and update the counter
for authors_list in data['author'].dropna():
    author_frequencies.update(authors_list)

# Determine the number of top authors you want to consider, e.g., top 100
num_top_authors = 100

# Get the most common authors
most_common_authors = author_frequencies.most_common(num_top_authors)

# Assuming 'most_common_authors' contains your top authors
top_authors = [author for author, count in most_common_authors]

# Dictionary to hold year distribution data for each top author
year_distributions = {}

for author in top_authors:
    # Filter data for the current author
    author_data = data[data['author'].apply(lambda x: author in x if isinstance(x, list) else False)]
    
    # Get year distribution for this author
    year_distributions[author] = author_data['year'].describe()

    
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


# Categories and counts for plotting
categories, counts = zip(*frequency_categories.items())

# Identify authors with 50+ publications
prolific_authors = [author for author, count in author_frequencies.items() if count >= 50]

# One-hot encode these authors
for author in prolific_authors:
    data[f'author_{author}'] = data['author'].apply(lambda x: author in x if isinstance(x, list) else False)

# Assuming each entry in the 'author' column is a list of authors
data['author_count'] = data['author'].apply(lambda x: len(x) if isinstance(x, list) else 0)

# TITLE and ABSTRACT COLUMNS

# Make Lower case
data['title_processed'] = data['title'].str.lower()

# Feature Extraction: TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=2000, use_idf=False)  # Limit features to 1000 for simplicity
title_tfidf = vectorizer.fit_transform(data['title_processed'])

# Convert to DataFrame
title_tfidf_df = pd.DataFrame(title_tfidf.toarray(), columns=vectorizer.get_feature_names_out())

data['title_processed'] = data['title'].str.lower()


# Lowercase Abstract 
""" filling na with no_abstarct decreases error by around 0.04 and use CountVectonizer
and using also lenght of abstarct as feature (together with countervectonizer) decreases error by 0.03"""

data['abstract_processed'] = data['abstract'].fillna('no_abstract').str.lower()

data['abstract_length'] = data['abstract_processed'].apply(len)

# Feature Extraction: TF-IDF for 'abstract'
abstract_vectorizer = CountVectorizer(stop_words='english', max_features=2000)  # Limit features to 1000
abstract_tfidf = abstract_vectorizer.fit_transform(data['abstract_processed'])

# Convert 'abstract' TF-IDF to DataFrame
abstract_tfidf_df = pd.DataFrame(abstract_tfidf.toarray(), columns=abstract_vectorizer.get_feature_names_out())

# NOW, LET'S DO RANDOM FOREST WITH ALL THE FEATURES TREATED

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd
import time

X = pd.concat([data.drop(['ENTRYTYPE','year', 'title', "editor", 'abstract', 'publisher', 'author', 'title_processed', 'abstract_processed'], axis=1),
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



# HYPERPARAMETER TUNING

#from sklearn.model_selection import RandomizedSearchCV
#from sklearn.ensemble import RandomForestRegressor

# Define the parameter grid to search
#param_grid = {
 #   'n_estimators': [100, 200, 300, 400, 500],
  #  'max_features': ['auto', 'sqrt'],
   # 'max_depth': [10, 20, 30, 40, 50, None],
    #'min_samples_split': [2, 5, 10],
    #'min_samples_leaf': [1, 2, 4],
    #'bootstrap': [True, False]
#}

# Initialize the Random Forest Regressor
#rf = RandomForestRegressor(random_state=42)

# Initialize the RandomizedSearchCV object
#rf_random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=100, cv=5, verbose=2, random_state=42, n_jobs=-1)

# Fit the random search model
#rf_random_search.fit(X_train, y_train)

# Print the best parameters found by RandomizedSearchCV
#print("Best parameters found: ", rf_random_search.best_params_)

# Evaluate the best model found on the test set
#best_model = rf_random_search.best_estimator_
#y_pred = best_model.predict(X_test)
#mae = mean_absolute_error(y_test, y_pred)
#print(f"Mean Absolute Error with best model: {mae}")

""" Result from hypertuning:  Best parameters found:  {'n_estimators': 200, 'min_samples_split': 5, 
'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': None, 'bootstrap': True}
Mean Absolute Error with best model: 3.4294981446638855"""

