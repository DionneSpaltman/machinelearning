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
# plt.figure(figsize=(10, 6))
# sns.histplot(data['year'], kde=False, color='blue', bins=30)
# plt.title('Distribution of Publication Years')
# plt.xlabel('Year')
# plt.ylabel('Frequency')
# plt.grid(True)
# plt.show()


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
# plt.figure(figsize=(15, 8))
# sns.boxplot(x='publisher', y='year', data=filtered_data)
# plt.xticks(rotation=45)
# plt.title('Distribution of Publication Years for Top Publishers')
# plt.show()


# Bar Plot
# mean_years.plot(kind='bar', figsize=(15, 8))
# plt.title('Mean Publication Year for Each Publisher')
# plt.xlabel('Publisher')
# plt.ylabel('Mean Publication Year')
# plt.xticks(rotation=90)
# plt.ylim(1940, mean_years.max() + 1)  # Set y-axis limits
# plt.show()

# Since there are 8200 NA from 65000 items, the decision is not as easy
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
# plt.figure(figsize=(10, 6))
# plt.bar(categories, counts, color='skyblue')
# plt.xlabel('Number of Publications')
# plt.ylabel('Number of Authors')
# plt.title('Distribution of Authors by Number of Publications')
# plt.show()

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


# Importing further Text-processing techniques
from sklearn.feature_extraction.text import HashingVectorizer


# TfidfVectorizer (n = 500)= 3.7
# Hashing Vectorizer (n = 1000) = 3.40
# Bert : Quitted after 2 1/2 hours


# TITLE and ABSTRACT COLUMNS

# Make Lower case
data['title_processed'] = data['title'].str.lower()

# Feature Extraction: TF-IDF
vectorizer = HashingVectorizer(n_features=1000)  # Limit features to 1000 for simplicity
title_tfidf = vectorizer.fit_transform(data['title_processed'])

# Convert to DataFrame
title_tfidf_df = pd.DataFrame(title_tfidf.toarray(), columns=[f'title_{i}' for i in range(1000)])
title_tfidf_df

# Lowercase Abstract
data['abstract_processed'] = data['abstract'].fillna('').str.lower()

# Feature Extraction: TF-IDF for 'abstract'
abstract_vectorizer = HashingVectorizer(n_features=1000)  # Limit features to 1000
abstract_tfidf = abstract_vectorizer.fit_transform(data['abstract_processed'])

# Convert 'abstract' TF-IDF to DataFrame
abstract_tfidf_df = pd.DataFrame(abstract_tfidf.toarray(), columns=[f'abstract{i}' for i in range(1000)])

import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from transformers import DistilBertTokenizer, DistilBertModel
import torch

# Load pre-trained DistilBERT tokenizer and model
# tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
# model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Function to get BERT embeddings for a given text
# def get_bert_embeddings(text):
#    tokens = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
#    with torch.no_grad():
#        outputs = model(**tokens)
#    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
#    return embeddings

# Process title and abstract
#data['title_processed'] = data['title'].str.lower()
#data['abstract_processed'] = data['abstract'].fillna('').str.lower()

# Get BERT embeddings for title and abstract
# data['title_bert_embeddings'] = data['title_processed'].apply(get_bert_embeddings)
# data['abstract_bert_embeddings'] = data['abstract_processed'].apply(get_bert_embeddings)

# Convert BERT embeddings to DataFrame
# title_bert_df = pd.DataFrame(data['title_bert_embeddings'].tolist(), columns=[f'title_bert_{i}' for i in range(768)])
# abstract_bert_df = pd.DataFrame(data['abstract_bert_embeddings'].tolist(), columns=[f'abstract_bert_{i}' for i in range(768)])

# Concatenate the DataFrames
# result_df = pd.concat([title_bert_df, abstract_bert_df], axis=1)



# NOW, LET'S DO RANDOM FOREST WITH ALL THE FEATURES TREATED

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
# import lightgbm as lgb


import pandas as pd
import time

X = pd.concat([data.drop(['year', 'title', 'abstract', 'publisher', 'author', 'title_processed', 'abstract_processed'], axis=1),
               title_tfidf_df , abstract_tfidf_df], axis=1).copy()
y = data['year']

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
# model = RandomForestRegressor(n_estimators=100, max_depth = 7, n_jobs=-1, random_state=42)
# model = GradientBoostingRegressor(learning_rate=0.1, n_estimators=100, max_depth=3, random_state=42) --> 4.20
# model = AdaBoostRegressor(n_estimators=50, learning_rate=1.0, random_state=42) --> 5.08
model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=7, random_state=42) 
# model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42) --> did not import library
# model = CatBoostRegressor(iterations=100, learning_rate=0.1, depth=6, random_state=42, logging_level='Silent') --> did not import library

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