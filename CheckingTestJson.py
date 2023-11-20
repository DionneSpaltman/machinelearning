import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

# WHOLE DATA

testdata = pd.read_json('test.json')
print(testdata.head())

# Basic info about data
print(testdata.info())

# Summary statistics for numerical fields
print(testdata.describe())



# ENTRYTYPE COLUMN


print(testdata['ENTRYTYPE'].unique())
# Only 3 types, probably good to use One-Hot Encoding

# One-hot encode the 'ENTRYTYPE' column
entrytype_dummies = pd.get_dummies(testdata['ENTRYTYPE'], prefix='entrytype')

# Concatenate the new columns with the original dataframe
testdata = pd.concat([testdata, entrytype_dummies], axis=1)

# Drop the original 'ENTRYTYPE' column
testdata.drop('ENTRYTYPE', axis=1, inplace=True)
print(testdata.info())

# Counting How Many item in each entrytype (used sum because dummy coded uses boolean
# this code counted how many 'true' for each column)
entrytype_counts = testdata[['entrytype_article', 'entrytype_inproceedings', 'entrytype_proceedings']].sum()
print(entrytype_counts)

# EDITOR COLUMN


# Count null values in the 'editor' column
null_count = testdata['editor'].isnull().sum()
print(f"Number of null values in 'editor': {null_count}")
# 64438 null values from 65914 data points. I guess.. Delete it?

testdata.drop('editor', axis=1, inplace=True)
print(testdata.info())


# PUBLISHER COLUMN


# Display unique values in the 'publisher' column
unique_publishers = testdata['publisher'].unique()
print(unique_publishers[:30])  # Display the first 30 unique values

# Count of unique values
print(f"Number of unique publishers: {testdata['publisher'].nunique()}")
# There are 120 Unique Publishers

# Display the frequency of each publisher
publisher_counts = testdata['publisher'].value_counts()
print(publisher_counts.head(20))  # Display the top 20 most frequent publishers

# Count null values in the 'publisher' column
null_count = testdata['publisher'].isnull().sum()
print(f"Number of null values in 'publisher': {null_count}")
# 8201 null values from 65914 Data Points


# AUTHOR COLUMN

# Flatten the list of authors
all_authors = set()
for authors_list in testdata['author'].dropna():
    all_authors.update(authors_list)

len(all_authors)
# Now all_authors contains all unique authors

from collections import Counter

# Initialize a Counter object to hold author frequencies
author_frequencies = Counter()

# Iterate through the author lists and update the counter
for authors_list in testdata['author'].dropna():
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