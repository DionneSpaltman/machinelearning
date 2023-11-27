


"""changes:
ABSTRACT: fill missing values with no_abstract + use countVectonizer + also variable len of abstarct
EDITOR: adding editor column with just number of editors

--> error in total decreased from 3.53 to 3.4189"""


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


# TITLE and ABSTRACT COLUMNS

# Make Lower case
data['title_processed'] = data['title'].str.lower()

# Feature Extraction: TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=500)  # Limit features to 500 for simplicity
title_tfidf = vectorizer.fit_transform(data['title_processed'])

# Convert to DataFrame
title_tfidf_df = pd.DataFrame(title_tfidf.toarray(), columns=vectorizer.get_feature_names_out())

# Lowercase Abstract 
""" filling na with no_abstarct decreases error by around 0.04 and use CountVectonizer
and using also lenght of abstarct as feature (together with countervectonizer) decreases error by 0.03"""


data['abstract_processed'] = data['abstract'].fillna('no_abstract').str.lower()

data['abstract_length'] = data['abstract_processed'].apply(len)

# Feature Extraction: TF-IDF for 'abstract'
abstract_vectorizer = CountVectorizer(stop_words='english', max_features=500)  # Limit features to 500
abstract_tfidf = abstract_vectorizer.fit_transform(data['abstract_processed'])

# Convert 'abstract' TF-IDF to DataFrame
abstract_tfidf_df = pd.DataFrame(abstract_tfidf.toarray(), columns=abstract_vectorizer.get_feature_names_out())


# NOW, LET'S DO RANDOM FOREST WITH ALL THE FEATURES TREATED

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pandas as pd
import time

#X = pd.concat(data.drop(['ENTRYTYPE', 'year', "title", "editor", "year", "publisher", "author", "abstract", 'abstract_processed']), axis=1).copy()
#X = pd.concat([data.drop(['ENTRYTYPE', 'year', "title", "editor", "year", "publisher", "author", "abstract"], axis=1)], axis=1).copy()
#X = pd.concat([data.drop(['ENTRYTYPE','year', 'title', "editor", 'abstract', 'publisher', 'author', 'title_processed', 'abstract_processed'], axis=1),
 #              title_tfidf_df, abstract_tfidf_df], axis=1).copy()
X = pd.concat([data.drop(['ENTRYTYPE','year', 'title', "editor", 'abstract', 'publisher', 'author', 'title_processed', 'abstract_processed'], axis=1),
               title_tfidf_df], axis=1).copy()

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




""" experimented with neural networks, but got error: 14.81 (also needed to instal keras and tenserflow with pip)"""
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
#from sklearn.metrics import mean_absolute_error
#from keras.models import Sequential
#from keras.layers import Dense

# Assuming X and y are already defined

# Splitting the dataset
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
#scaler = StandardScaler()
#X_train_scaled = scaler.fit_transform(X_train)
#X_test_scaled = scaler.transform(X_test)

# Build the neural network model
#model = Sequential()
#model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
#model.add(Dense(32, activation='relu'))
#model.add(Dense(1, activation='linear'))

# Compile the model
#model.compile(loss='mean_absolute_error', optimizer='adam')

# Train the model
#model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_data=(X_test_scaled, y_test))

# Predict on the testing set
#y_pred_nn = model.predict(X_test_scaled)

# Calculate Mean Absolute Error
#mae_nn = mean_absolute_error(y_test, y_pred_nn)
#print(f"Neural Network Mean Absolute Error: {mae_nn}")

