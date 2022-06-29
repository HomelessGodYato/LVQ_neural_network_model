import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# script for entry data analysis
# TODO: whatever you want

df = pd.read_csv(
    'D:/desktop/Programming/Python/AI_ML/Neural networks/University Project/LVQ_NN/german_credit_data.csv')

sns.countplot(data=df, x="kredit")
plt.xlabel("Credit risk")
plt.ylabel("Records count")
plt.show()