import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    credit_data = pd.read_csv('D:/desktop/Programming/Python/AI_ML/Neural networks/University Project/LVQ_NN/german_credit_data.csv')
    _class = credit_data['kredit']
    del credit_data['Id']


if __name__ == '__main__':
    main()