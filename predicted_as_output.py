import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

complete_data = pd.read_json('input/train.json')
data = pd.read_json('input/train.json')
test_data = pd.read_json('input/test.json')

# -------------------------------------------------Feature: entrytype-------------------------------------------------#
# It has 3 unique features
# So we one-hot encode the 'ENTRYTYPE' column
all_entrytypes = pd.concat([data['ENTRYTYPE'], test_data['ENTRYTYPE']], axis=0)

# One-hot encode 'ENTRYTYPE' and concatenate with the original DataFrame
entrytype_dummies = pd.get_dummies(all_entrytypes, prefix='entrytype')

# Drop the original 'ENTRYTYPE' column
data.drop('ENTRYTYPE', axis=1, inplace=True)
test_data.drop('ENTRYTYPE', axis=1, inplace=True)

entrytype_dummies.info()



# -------------------------------------------------Feature: publisher-------------------------------------------------#
# Impute the NA as 'unknown'
all_publishers = pd.concat([data['publisher'], test_data['publisher']], axis=0)
all_publishers.fillna('Unknown', inplace=True)

# One-hot encode the 'publisher' column
publisher_dummies = pd.get_dummies(all_publishers, prefix='publisher')
publisher_dummies.info()

# Drop the original 'publisher' column
data.drop('publisher', axis=1, inplace=True)
test_data.drop('publisher', axis=1, inplace=True)

# -------------------------------------------------Feature: author-------------------------------------------------#
# Make it a set to get unique authors
complete_authors = pd.concat([data['author'], test_data['author']], axis=0)

all_authors = set()
for authors_list in complete_authors.dropna():
    all_authors.update(authors_list)
    
# Count how many times each author appears
author_frequencies = Counter()

for authors_list in complete_authors.dropna():
    author_frequencies.update(authors_list)

# Set Prolific Authors (After trials, decided on 25+ Publications)    
prolific_authors = [author for author, count in author_frequencies.items() if count >= 25]

author_dummies = data.copy()
author_dummies.drop(['editor','title','year','abstract'], axis=1, inplace=True)

# One-hot encode these authors
for author in prolific_authors:
    author_dummies[f'author_{author}'] = author_dummies['author'].apply(lambda x: author in x if isinstance(x, list) else False)

author_dummies.drop('author', axis=1, inplace=True)
author_dummies = author_dummies.copy()
author_dummies.info()

# -------------------------------------------------New feature: author count-------------------------------------------------#

# Could be a good predictor: newer publications are more collaborative
author_count = data['author'].apply(lambda x: len(x) if isinstance(x, list) else 0)
author_count.info()

# Drop the original 'author' column
data.drop('author', axis=1, inplace=True)

# FEATURE: TITLE

# Make Title Lower case
title_lower_train = data['title'].str.lower()

# Process the title with TF-IDF. Works better than hashing or count vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
title_tfidf_train = vectorizer.fit_transform(title_lower_train)

# Convert to DataFrame to be used in the prediction
title_processed_train = pd.DataFrame(title_tfidf_train.toarray(), columns=vectorizer.get_feature_names_out())
title_processed_train.info()

# FEATURE: ABSTRACT

# Make lowercase for further processing
abstract_lower_train = data['abstract'].fillna('no_abstract').str.lower()

# Abstract - COUNT VECTORIZER
abstract_vectorizer = CountVectorizer(stop_words='english', max_features=2000)
abstract_processed_train = abstract_vectorizer.fit_transform(abstract_lower_train)

# Convert to DataFrame to be used in the prediction
abstract_processed_train = pd.DataFrame(abstract_processed_train.toarray(), columns=abstract_vectorizer.get_feature_names_out())
abstract_processed_train.info()


# NEW FEATURE: LENGTH OF ABSTRACT

abstract_length = abstract_lower_train.apply(len)
abstract_length.info()

# NEW FEATURE: NUMBER OF EDITORS

editor_count = complete_data['editor'].apply(lambda x: len(x) if isinstance(x, list) else 0)
editor_count.info()

# FEATURE: EDITOR

# Replace missing values with with 'Unknown'
complete_editors = pd.concat([data['editor'], test_data['editor']], axis=0)
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

data.info()
editor_dummies = data.copy()
editor_dummies.info()

for editor in frequent_editors:
    editor_dummies[f'editor_{editor}'] = editor_dummies['editor'].apply(lambda x: editor in x if isinstance(x, list) else editor == x)

editor_dummies.drop(['title','editor','year','abstract'], axis=1, inplace=True)
editor_dummies.info()

# Drop the original 'editor' column
data.drop('editor', axis=1, inplace=True)


###
###
###

# CLEANING TEST.JSON


# -------------------------------------------------New feature: author count-------------------------------------------------#

# Could be a good predictor: newer publications are more collaborative
test_author_count = test_data['author'].apply(lambda x: len(x) if isinstance(x, list) else 0)
test_data.drop('author', axis=1, inplace=True)
test_author_count.info()

# FEATURE: TITLE

# Make Title Lower case
test_title_lower = test_data['title'].str.lower()

# Process the title with TF-IDF using the words from training data
test_title_tfidf = vectorizer.transform(test_title_lower)
test_title_processed = pd.DataFrame(test_title_tfidf.toarray(), columns=vectorizer.get_feature_names_out())

test_title_processed.info()

# FEATURE: ABSTRACT

# Make Abstract Lower case for test data
test_abstract_lower = test_data['abstract'].fillna('no_abstract').str.lower()

# Transform the abstract using the fitted CountVectorizer from training data
test_abstract_count = abstract_vectorizer.transform(test_abstract_lower)
test_abstract_processed = pd.DataFrame(test_abstract_count.toarray(), columns=abstract_vectorizer.get_feature_names_out())

test_abstract_processed.info()


# NEW FEATURE: LENGTH OF ABSTRACT

test_abstract_length = test_abstract_lower.apply(len)
test_abstract_length.info()


# NEW FEATURE: NUMBER OF EDITORS

test_editor_count = test_data['editor'].apply(lambda x: len(x) if isinstance(x, list) else 0)
test_data.drop('editor', axis=1, inplace=True)
test_editor_count.info()

###
###
###

# MODEL: RANDOM FOREST

# Dataset

X = pd.concat([entrytype_dummies.iloc[:len(data),:], publisher_dummies.iloc[:len(data),:], author_dummies.iloc[:len(data),:], author_count, title_processed_train, abstract_processed_train, abstract_length, editor_dummies.iloc[:len(data),:], editor_count], axis=1).copy()
y = data['year']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

# Initialize the Random Forest Regressor
model = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=0)

# Train the model
model.fit(X_train, y_train)

# Predict on the validation set
y_pred = model.predict(X_val)

# Calculate Mean Absolute Error on the training data
mae = mean_absolute_error(y_val, y_pred)
print(f"Mean Absolute Error on the validation set: {mae}")

# PREDICTING ON THE TEST DATA
X_test = pd.concat([entrytype_dummies.iloc[:len(test_data),:], publisher_dummies.iloc[:len(test_data),:], author_dummies.iloc[:len(test_data),:], test_author_count, test_title_processed, test_abstract_processed, test_abstract_length, editor_dummies.iloc[:len(test_data),:], test_editor_count], axis=1).copy()
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

test_predictions = model.predict(X_test)

# OUTPUT

# Output only the year:
year_predictions_df = pd.DataFrame({'year': test_predictions})

# Output to predicted.json file
year_predictions_df.to_json("predictions/testpredictednotrounder.json", orient='records', indent=2)

'''
ADJUSTED FOR BIAS

# Calculate the errors
errors = y_val - y_pred

# Calculate the mean error
mean_error = errors.mean()
mean_error

# Adjust predictions by the mean error
adjusted_predictions = test_predictions - mean_error

# Round predictions to the nearest integer after adjustment
final_predictions_rounded = np.round(adjusted_predictions).astype(int)

# Assign the adjusted and rounded predictions
test_data['year'] = final_predictions_rounded

# OUTPUT

# Output only the year:
year_predictions_df = pd.DataFrame({'year': final_predictions_rounded})

# Output to predicted.json file
year_predictions_df.to_json("predictions/testpredicted.json", orient='records', indent=2)
'''