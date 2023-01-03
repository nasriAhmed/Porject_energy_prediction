import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import dataset

data= pd.read_csv("./../data/data_energy.csv")

#show 10 first lignes
print(data.head(10))
print(data.describe())
print(data.info())