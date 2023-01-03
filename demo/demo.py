import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import dataset

data= pd.read_csv("./../data/data_energy.csv")

#show 10 first lignes
print(data.head(10))

#show describe data
print(data.describe())

#show information data
print(data.info())