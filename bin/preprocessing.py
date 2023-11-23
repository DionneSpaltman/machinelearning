"""
This document is for preprocessing purposes.
- Describing the data 
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------Whole data-------------------------------------------------#

data = pd.read_json('input/train.json')
print(data.head())

# Basic info about data
print(data.info())

# Summary statistics for numerical fields
print(data.describe())

# -------------------------------------------------Year column-------------------------------------------------#

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


# -------------------------------------------------Entrytype column-------------------------------------------------#

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


# -------------------------------------------------Editor Column-------------------------------------------------#

# Count null values in the 'editor' column
null_count = data['editor'].isnull().sum()
print(f"Number of null values in 'editor': {null_count}")
# 64438 null values from 65914 data points. I guess.. Delete it?

data.drop('editor', axis=1, inplace=True)
print(data.info())


# -------------------------------------------------Publisher column-------------------------------------------------#

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