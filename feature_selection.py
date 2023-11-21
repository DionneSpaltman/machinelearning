import pandas as pd
import numpy as np
import json
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Load the dataset from the provided URL
data = pd.DataFrame.from_records(json.load(open('train.json'))).fillna("")

# Specify the target variable name (replace 'target_name' with the actual name)
target_name = 'year'

# Extract the feature columns and target variable
X = data.drop(columns=[target_name])
y = data[target_name]

# Check data types
print("Data types in X:", X.dtypes)
print("Data type of y:", y.dtype)

# Define a list of categorical columns
categorical_columns = ['ENTRYTYPE', 'title', 'editor', 'publisher', 'author', 'abstract']

# Convert each categorical column to strings and handle lists if present
for col in categorical_columns:
    X[col] = X[col].apply(lambda x: ','.join(map(str, x)) if isinstance(x, list) else str(x))

# Create a column transformer for one-hot encoding
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_columns)
    ],
    remainder='passthrough'
)

# Apply one-hot encoding to the features
X_encoded = preprocessor.fit_transform(X)

# Check data types after preprocessing
print("Data types in X_encoded:", type(X_encoded))

# Select the top k features based on f-regression
k = 3  # Number of features to select
selector = SelectKBest(score_func=f_regression, k=k)
X_new = selector.fit_transform(X_encoded, y)

# Get the indices of the selected features
selected_feature_indices = selector.get_support(indices=True)

# Print the selected feature names
selected_feature_names = X.columns[selected_feature_indices]
print("Selected feature names:", selected_feature_names)
