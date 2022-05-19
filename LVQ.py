import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, datasets, linear_model
from sklearn.model_selection import train_test_split,KFold
import seaborn as sns

class LVQ:

    def load_data(self,file_name):
        """
        This function loads the dataset.
        """
        data = pd.read_csv(file_name)
        return data


nn = LVQ()
data = nn.load_data('../LVQ_NN/german_credit_data.csv')
print(data)
