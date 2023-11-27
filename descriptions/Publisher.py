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
from sklearn.feature_extraction.text import HashingVectorizer

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
# 30 publishers = MAE: 4.548
# 50 publishers = MAE: 4.548
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

publisher_processed = data['publisher'].fillna('Unknown', inplace=True)

# One-hot encoding of the 'publisher' column
# publisher_dummies = pd.get_dummies(data['publisher'], prefix='publisher')

# Join the one-hot encoded columns back to the original DataFrame
# data = pd.concat([data, publisher_dummies], axis=1)
# print(data)

publisher_vectorizer = HashingVectorizer(n_features=1000)  # Limit features to 1000
publisher_hash = publisher_vectorizer.fit_transform(data['publisher_processed '])


# Convert 'abstract' TF-IDF to DataFrame
publisher_hash _df = pd.DataFrame(publisher_hash.toarray(), columns=[f'publisher{i}' for i in range(1000)])