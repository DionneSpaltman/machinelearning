import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
"""
In this file you'll find 
- Feature engineering 
- Model: random forest (we deleted all of our experiments so this file is clean)
- Submitting the prediction to a json file
"""

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