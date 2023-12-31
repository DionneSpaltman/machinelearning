import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

'''
In this file, you will find the main code, followed by codes for data visualizations.
The code for data visualization starts on line 238
'''

# Data is loaded (from the input folder)
complete_data = pd.read_json('input/train.json')
train = pd.read_json('input/train.json')
test_data = pd.read_json('input/test.json')

# -------------------------------------------------Feature: entrytype-------------------------------------------------#
# Entrytype has three unique options. We decided to one-hot encode it 
all_entrytypes = pd.concat([train['ENTRYTYPE'], test_data['ENTRYTYPE']], axis=0)

# One-hot encode 'ENTRYTYPE' 
entrytype_dummies = pd.get_dummies(all_entrytypes, prefix='entrytype')

# Drop the original 'ENTRYTYPE' column
train.drop('ENTRYTYPE', axis=1, inplace=True)
test_data.drop('ENTRYTYPE', axis=1, inplace=True)

entrytype_dummies.info()

# -------------------------------------------------Feature: publisher-------------------------------------------------#
# Impute the NA as 'unknown'
all_publishers = pd.concat([train['publisher'], test_data['publisher']], axis=0)
all_publishers.fillna('Unknown', inplace=True)

# One-hot encode the 'publisher' column
publisher_dummies = pd.get_dummies(all_publishers, prefix='publisher')
publisher_dummies.info()

# Drop the original 'publisher' column
train.drop('publisher', axis=1, inplace=True)
test_data.drop('publisher', axis=1, inplace=True)

# -------------------------------------------------Feature: author-------------------------------------------------#
# Make it a set to get unique authors
complete_authors = pd.concat([train['author'], test_data['author']], axis=0)
all_authors = set()
for authors_list in complete_authors.dropna():
    all_authors.update(authors_list)
    
# Count how many times each author appears
author_frequencies = Counter()
for authors_list in complete_authors.dropna():
    author_frequencies.update(authors_list)

# Set Prolific Authors (After trials, decided on 25+ Publications)    
prolific_authors = [author for author, count in author_frequencies.items() if count >= 25]

author_dummies = train.copy()
author_dummies.drop(['editor','title','year','abstract'], axis=1, inplace=True)

# One-hot encode these authors
for author in prolific_authors:
    author_dummies[f'author_{author}'] = author_dummies['author'].apply(lambda x: author in x if isinstance(x, list) else False)

author_dummies.drop('author', axis=1, inplace=True)
author_dummies = author_dummies.copy()
author_dummies.info()

# -------------------------------------------------New feature: author count-------------------------------------------------#
# Could be a good predictor: newer publications are more collaborative
author_count = train['author'].apply(lambda x: len(x) if isinstance(x, list) else 0)
author_count.info()

# Drop the original 'author' column
train.drop('author', axis=1, inplace=True)

# -------------------------------------------------Feature: title-------------------------------------------------#
# Make Title Lower case
title_lower_train = train['title'].str.lower()

# Process the title with TF-IDF. Works better than hashing or count vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
title_tfidf_train = vectorizer.fit_transform(title_lower_train)

# Convert to DataFrame to be used in the prediction
title_processed_train = pd.DataFrame(title_tfidf_train.toarray(), columns=vectorizer.get_feature_names_out())
title_processed_train.info()

# -------------------------------------------------Feature: abstract-------------------------------------------------#
# Make lowercase for further processing
abstract_lower_train = train['abstract'].fillna('no_abstract').str.lower()

# Abstract - COUNT VECTORIZER
abstract_vectorizer = CountVectorizer(stop_words='english', max_features=1000)
abstract_processed_train = abstract_vectorizer.fit_transform(abstract_lower_train)

# Convert to DataFrame to be used in the prediction
abstract_processed_train = pd.DataFrame(abstract_processed_train.toarray(), columns=abstract_vectorizer.get_feature_names_out())
abstract_processed_train.info()

# -------------------------------------------------New feature: Length of Abstract-------------------------------------------------#
abstract_length = abstract_lower_train.apply(len)
abstract_length.info()

# -------------------------------------------------New feature: Number of Editors-------------------------------------------------#
editor_count = complete_data['editor'].apply(lambda x: len(x) if isinstance(x, list) else 0)
editor_count.info()

# -------------------------------------------------Feature: Editor-------------------------------------------------#
# Replace missing values with with 'Unknown'
complete_editors = pd.concat([train['editor'], test_data['editor']], axis=0)
complete_editors.fillna('Unknown', inplace=True)

# Make a set of unique editors
all_editors = set()
for editors_list in complete_editors:
    if isinstance(editors_list, list):
        all_editors.update(editors_list)
    else:
        all_editors.add(editors_list)

# Calculate editor frequencies
editor_frequencies = Counter()
for editors_list in complete_editors:
    if isinstance(editors_list, list):
        editor_frequencies.update(editors_list)
    else:
        editor_frequencies.update([editors_list])

# One-hot encode editors which appears 5 or more times
threshold = 5
frequent_editors = [editor for editor, count in editor_frequencies.items() if count >= threshold]

train.info()
editor_dummies = train.copy()
editor_dummies.info()

for editor in frequent_editors:
    editor_dummies[f'editor_{editor}'] = editor_dummies['editor'].apply(lambda x: editor in x if isinstance(x, list) else editor == x)

editor_dummies.drop(['title','editor','year','abstract'], axis=1, inplace=True)
editor_dummies.info()

# Drop the original 'editor' column
train.drop('editor', axis=1, inplace=True)

# CLEANING TEST.JSON
# -------------------------------------------------New feature: author count-------------------------------------------------#
# Could be a good predictor: newer publications are more collaborative
test_author_count = test_data['author'].apply(lambda x: len(x) if isinstance(x, list) else 0)
test_data.drop('author', axis=1, inplace=True)
test_author_count.info()

# -------------------------------------------------New feature: Title-------------------------------------------------#
# Make Title Lower case
test_title_lower = test_data['title'].str.lower()

# Process the title with TF-IDF using the words from training data
test_title_tfidf = vectorizer.transform(test_title_lower)
test_title_processed = pd.DataFrame(test_title_tfidf.toarray(), columns=vectorizer.get_feature_names_out())

test_title_processed.info()

# -------------------------------------------------Feature: abstract-------------------------------------------------#
# Make Abstract Lower case for test data
test_abstract_lower = test_data['abstract'].fillna('no_abstract').str.lower()

# Transform the abstract using the fitted CountVectorizer from training data
test_abstract_count = abstract_vectorizer.transform(test_abstract_lower)
test_abstract_processed = pd.DataFrame(test_abstract_count.toarray(), columns=abstract_vectorizer.get_feature_names_out())

test_abstract_processed.info()

# -------------------------------------------------New feature: Length of Abstract-------------------------------------------------#
test_abstract_length = test_abstract_lower.apply(len)
test_abstract_length.info()

# -------------------------------------------------New feature: Number of Editors-------------------------------------------------#
test_editor_count = test_data['editor'].apply(lambda x: len(x) if isinstance(x, list) else 0)
test_data.drop('editor', axis=1, inplace=True)
test_editor_count.info()

# -------------------------------------------------Model: Random Forest-------------------------------------------------#
# Validation
X = pd.concat([entrytype_dummies.iloc[:len(train),:],
               publisher_dummies.iloc[:len(train),:],
               author_dummies.iloc[:len(train),:],
               author_count,
               title_processed_train,
               abstract_processed_train,
               abstract_length,
               editor_dummies.iloc[:len(train),:],
               editor_count], axis=1).copy()
y = train['year']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0, stratify=train['year'])

# Initialize the Random Forest Regressor
model = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=0)

# Train the model
model.fit(X_train, y_train)

# Predict on the validation set
y_pred = model.predict(X_val)

# Calculate Mean Absolute Error on the training data
mae = mean_absolute_error(y_val, y_pred)
print(f"Mean Absolute Error on the validation set: {mae}")

# -------------------------------------------------Predicting on test data -------------------------------------------------#
test = pd.concat([entrytype_dummies.iloc[len(train):,:],
                  publisher_dummies.iloc[len(train):,:],
                  author_dummies.iloc[len(train):,:],
                  test_author_count,
                  test_title_processed,
                  test_abstract_processed,
                  test_abstract_length,
                  editor_dummies.iloc[len(train):,:],
                  test_editor_count], axis=1).copy()
test = test.reindex(columns=X_train.columns, fill_value=0)
test.fillna(0, inplace=True)

pred = model.predict(test)

if test.columns.duplicated().any():
    print("Duplicate columns found: ", test.columns[test.columns.duplicated()])
    test = test.loc[:,~test.columns.duplicated()]

year_predicted_df = pd.DataFrame(pred, columns=['year'])

# Save to a JSON file
year_predicted_df.to_json('predicted.json', orient='records', indent=2)

# -------------------------------------------------PLOTS-------------------------------------------------#

# APPENDIX 1: DISTRIBUTION OF YEAR

x = complete_data
print(x.describe())
print(x.columns)
x_year = x["year"]
print(x_year)

counts_df = x['year'].value_counts().reset_index()
counts_df.columns = ['Year', 'Count']

print(counts_df)
print(x.shape)
print(x.info())
print(x.head())
print(x.isnull().sum()/65914)

# Plotting the distribution of year
plt.hist(x_year, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')
sns.kdeplot(x_year, color='red', label='PDF')
plt.xlabel('Year')
plt.ylabel('Frequency')
plt.title('Distribution target variable')
plt.show()

# APPENDIX 2:

# Display the frequency of each publisher
publisher_counts = complete_data['publisher'].value_counts()

# Calculate mean year for each publisher
mean_years = complete_data.groupby('publisher')['year'].mean().sort_values()
mean_years

# Filter out to include only the top N publishers for a clearer plot
top_publishers = publisher_counts.index[:20]
top_publishers

filtered_data = complete_data[complete_data['publisher'].isin(top_publishers)]
filtered_data

# Create a box plot
plt.figure(figsize=(10, 5))
sns.boxplot(x='publisher', y='year', data=filtered_data)
plt.title('Distribution of Publication Years for Top Publishers')
plt.xticks(rotation=60, ha='right')
plt.tight_layout()
plt.show()

# APPENDIX 3: TOP PUBLISHERS

# Calculate the count of publications for each publisher
publisher_counts = complete_data['publisher'].value_counts()

# Select the top 20 publishers
top_publishers = publisher_counts.head(20)

# Create a bar plot
plt.figure(figsize=(10, 6))
sns.barplot(x=top_publishers.index, y=top_publishers.values, palette="viridis")
plt.title('Number of Publications by Top 20 Publishers')
plt.xlabel('Publisher')
plt.ylabel('Number of Publications')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# APPENDIX 4: Distribution of Authors by Number of Publications

# Flatten the list of authors
all_authors = set()
for authors_list in complete_data['author'].dropna():
    all_authors.update(authors_list)
len(all_authors)

# Calculation of author frequencies
author_frequencies = Counter()
for authors_list in complete_data['author'].dropna():
    author_frequencies.update(authors_list)
author_frequencies

# Exploration of top publishers
num_top_authors = 100
most_common_authors = author_frequencies.most_common(num_top_authors)
for author, count in most_common_authors:
    print(f"{author}: {count}")
top_authors = [author for author, count in most_common_authors]

# Exploration of the year distributions for top authors
year_distributions = {}
for author in top_authors:
    author_data = complete_data[complete_data['author'].apply(lambda x: author in x if isinstance(x, list) else False)]
    year_distributions[author] = author_data['year'].describe()

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

categories, counts = zip(*frequency_categories.items())

# Create a bar plot to illustrate the relationship between top authors and number of publications 
plt.figure(figsize=(10, 6))
plt.bar(categories, counts, color='skyblue')
plt.xlabel('Number of Publications')
plt.ylabel('Number of Authors')
plt.title('Distribution of Authors by Number of Publications')
plt.show()

# FEATURE IMPORTANCES

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
plt.xticks(rotation=45, ha='right')
plt.show()