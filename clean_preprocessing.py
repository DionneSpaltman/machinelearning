import time
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

complete_data = pd.read_json('train.json')
data = pd.read_json('train.json')

# ENTRYTYPE

# One-hot encode the 'ENTRYTYPE' column
entrytype_dummies = pd.get_dummies(data['ENTRYTYPE'], prefix='entrytype')

# Drop the original 'ENTRYTYPE' column
data.drop('ENTRYTYPE', axis=1, inplace=True)

# PUBLISHER

# Impute the NA as 'unknown_publisher'
data['publisher'].fillna('Unknown', inplace=True)

# One-hot encoding of the 'publisher' column
publisher_dummies = pd.get_dummies(data['publisher'], prefix='publisher')
publisher_dummies

# Drop the original 'publisher' column
data.drop('publisher', axis=1, inplace=True)

# AUTHOR

# Flatten the list of authors
all_authors = set()
for authors_list in data['author'].dropna():
    all_authors.update(authors_list)
    
# Initialize a Counter object to hold author frequencies
author_frequencies = Counter()

# Iterate through the author lists and update the counter
for authors_list in data['author'].dropna():
    author_frequencies.update(authors_list)

# Set Prolific Authors (25+ Publications)    
prolific_authors = [author for author, count in author_frequencies.items() if count >= 25]

author_dummies = data.copy()
author_dummies.info()

# One-hot encode these authors
for author in prolific_authors:
    author_dummies[f'author_{author}'] = author_dummies['author'].apply(lambda x: author in x if isinstance(x, list) else False)

author_dummies.drop(['editor','title','year','author','abstract'], axis=1, inplace=True)
author_dummies = author_dummies.copy()

# Dropping Author from data
data.drop('author', axis=1, inplace=True)
data.columns

# AUTHOR COUNT (NEW VARIABLE)

# Assuming each entry in the 'author' column is a list of authors
author_count = complete_data['author'].apply(lambda x: len(x) if isinstance(x, list) else 0)
author_count

# TITLE

# Make Title Lower case
title_lower = data['title'].str.lower()

# Feature Extraction: TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=500)  # Limit features to 500
title_tfidf = vectorizer.fit_transform(title_lower)

# Convert to DataFrame
title_tfidf_df = pd.DataFrame(title_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
title_processed = title_tfidf_df.copy()

# Drop title from data
data.drop('title', axis=1, inplace=True)
data.columns

# ABSTRACT

abstract_lower = data['abstract'].fillna('').str.lower()

# Feature Extraction: TF-IDF for 'abstract'
abstract_vectorizer = TfidfVectorizer(stop_words='english', max_features=500)  # Limit features to 500
abstract_tfidf = abstract_vectorizer.fit_transform(abstract_lower)

# Convert 'abstract' TF-IDF to DataFrame
abstract_tfidf = pd.DataFrame(abstract_tfidf.toarray(), columns=abstract_vectorizer.get_feature_names_out())
abstract_processed = abstract_tfidf.copy()

# Drop abstract from data
data.drop('abstract', axis=1, inplace=True)
data.columns

# Abstract Length (new column)

# LENGTH OF ABSTRACT (NEW VARIABLE)

abstract_length = abstract_lower.apply(len)
abstract_length

# EDITOR (will figure out later, now drop first)

# Replace NaN with a placeholder
data['editor'].fillna('Unknown', inplace=True)

# Flatten the list of editors
all_editors = set()
for editors_list in data['editor']:
    if isinstance(editors_list, list):
        all_editors.update(editors_list)
    else:
        all_editors.add(editors_list)

# Calculate editor frequencies
editor_frequencies = Counter()
for editors_list in data['editor']:
    if isinstance(editors_list, list):
        editor_frequencies.update(editors_list)
    else:
        editor_frequencies.update([editors_list])

# Identify frequent editors
threshold = 5  # or any other number you deem appropriate
frequent_editors = [editor for editor, count in editor_frequencies.items() if count >= threshold]

editor_dummies = data.copy()
editor_dummies.info()

# One-hot encode frequent editors
for editor in frequent_editors:
    editor_dummies[f'editor_{editor}'] = editor_dummies['editor'].apply(lambda x: editor in x if isinstance(x, list) else editor == x)

# Drop the original 'editor' column
data.drop('editor', axis=1, inplace=True)
editor_dummies.drop('editor', axis=1, inplace=True)
editor_dummies.drop('year', axis=1, inplace=True)



# RANDOM FOREST

X = pd.concat([entrytype_dummies, publisher_dummies, author_dummies, author_count, title_processed, abstract_processed, abstract_length, editor_dummies], axis=1).copy()
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
