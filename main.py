import numpy as np

import pandas as pd
from pandas.plotting import scatter_matrix

import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('D:/desktop/Programming/Python/AI_ML/Neural networks/University Project/LVQ_NN/german_credit_data.csv')
# plt.figure(figsize=(15,9))
# sns.heatmap(df.corr(), cmap='Blues', annot=True)
# plt.show()
corr_matrix = df.corr()

print(corr_matrix['kredit'].sort_values(ascending=False))