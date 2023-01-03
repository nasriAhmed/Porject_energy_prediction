import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
#import seaborn as sns
#sns.set(font_scale = 2)

# Splitting data into training and testing
from sklearn.model_selection import train_test_split
print("******")
print("Import Dataset")
print("******")
data= pd.read_csv("./../data/data_energy.csv")
print(data)

#show 10 first lignes
print(data.head())

#show describe data
#print(data.describe())

#show information data
#print(data.info())

print("******")
print("Cleaning Data")
print("******")
## Replace all occurrences of Not Available with numpy not a number
data = data.replace({'Not Available',np.nan})
print(data)

## Iterate through the columns
for col in list(data.columns):
    # Select columns that should be numeric
    if ('ft²' in col or 'kBtu' in col or 'Metric Tons CO2e' in col or 'kWh' in
        col or 'therms' in col or 'gal' in col or 'Score' in col):
        #Convert the data type to float
         print(data[col])
        #data[col] = data[col].astype(float)

#Missing values
# Function to calculate missing values by column
    def missing_values_table(df):
        # Total missing values
        mis_val = df.isnull().sum()

        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)

        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
            columns={0: 'Missing Values', 1: '% of Total Values'})

        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
            '% of Total Values', ascending=False).round(1)

        # Print some summary information
        print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
                                                                  "There are " + str(
            mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")

        # Return the dataframe with missing information
        return mis_val_table_ren_columns
print(missing_values_table(data))
print(data)
# Get the columns with > 50% missing
missing_df = missing_values_table(data);
missing_columns = list(missing_df[missing_df['% of Total Values'] > 50].index)
print('We will remove %d columns.' % len(missing_columns))

# Drop the columns
data = data.drop(columns = list(missing_columns))
#print(data)

print("******")
print("Exploratory Data Analysis")
print("******")

print("importer matplotlib.pyplot en tant que plt")
# Rename the score
figsize(12, 12)
data = data.rename(columns = {'ENERGY STAR Score': 'score'})

# Histogram of the Energy Star Score
plt.style.use('fivethirtyeight')
plt.hist(data['score'].dropna(), bins = 100, edgecolor = 'k');
plt.xlabel('Score'); plt.ylabel('Nombre de bâtiments');
plt.title('Répartition des scores Energy Star');
#plt.show()


# Find all correlations and sort
correlations_data = data.corr()['score'].sort_values()

# Print the most negative correlations
print(correlations_data.head(15), '\n')

# Print the most positive correlations
print(correlations_data.tail(15))

print("Split Into Training and Testing Sets")
no_score = features[features['score'].isna()]
score = features[features['score'].notnull()]

print(no_score.shape)
print(score.shape)

# Separate out the features and targets
features = score.drop(columns='score')
targets = pd.DataFrame(score['score'])

# Replace the inf and -inf with nan (required for later imputation)
features = features.replace({np.inf: np.nan, -np.inf: np.nan})

# Split into 70% training and 30% testing set
X, X_test, y, y_test = train_test_split(features, targets, test_size = 0.3, random_state = 42)

print(X.shape)
print(X_test.shape)
print(y.shape)
print(y_test.shape)