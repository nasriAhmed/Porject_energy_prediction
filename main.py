import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data= pd.read_csv("data/data_energy.csv")
print(data.info())
print(data.head(10))
print(data.describe())

data= data.replace({'Not available' : np.nan})
#data = data.replace('Not available', np.nan)
df=pd.DataFrame(data)
print(df)
#for col in list(data.columns):
    #if('ft' in col or 'ktbu' in col):
        #data[col] = data[col].astype(float)

#mis_vall=data.column.isnull().sum()
#print(mis_vall)
# Create a list of buildings with more than 100 measurements

# Create a list of buildings with more than 100 measurements
types = data.dropna(subset=['score'])
types = types['Largest Property Use Type'].value_counts()
types = list(types[types.values > 100].index)

# Plot of distribution of scores for building categories
figsize(12, 10)

# Plot each building
for b_type in types:
    # Select the building type
    subset = data[data['Largest Property Use Type'] == b_type]

    # Density plot of Energy Star scores
    sns.kdeplot(subset['score'].dropna(),
                label=b_type, shade=False, alpha=0.8);

# label the plot
plt.xlabel('Energy Star Score', size=20);
plt.ylabel('Density', size=20);
plt.title('Density Plot of Energy Star Scores by Building Type', size=28);