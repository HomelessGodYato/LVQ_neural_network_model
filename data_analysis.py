import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
"""
Wczytywanie danych z pliku csv
"""
df = pd.read_csv(
    'D:/desktop/Programming/Python/AI_ML/Neural networks/University Project/LVQ_NN/german_credit_data.csv')

"""
Ogólne informacje o danych
"""

"""
Rysowanie wykresu
"""
sns.countplot(data=df, x="kredit")
plt.xlabel("Ryzyko kredytowe")
plt.ylabel("Liczba rekordów")
plt.show()