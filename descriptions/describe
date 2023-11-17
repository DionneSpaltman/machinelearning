#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%%
x = pd.read_json('../train.json')
print(x.describe())
print(x.columns)
x_year = x["year"]
print(x_year)

counts_df = x['year'].value_counts().reset_index()
counts_df.columns = ['Year', 'Count']
print(counts_df)
#%%
print(x.shape)
print(x.info())
print(x.head())
print(x.isnull().sum()/65914)
# %%
# Plotting the distribution of year
plt.hist(x_year, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')
sns.kdeplot(x_year, color='red', label='PDF')

# Add labels and a title
plt.xlabel('Year')
plt.ylabel('Frequency')
plt.title('Distribution Plot')

# Show the plot
plt.show()

# %%
