from math import sqrt, pow
from random import randrange

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing



class LVQ:

    def load_data(self, file_name):
        """
        This function loads the dataset.
        """
        data = pd.read_csv(file_name)
        # columns = ["current_account",
        #            "duration", "behavior",
        #            "purpose", "amount",
        #            "savings",
        #            "employment_time",
        #            "installment_rate",
        #            "sex/family",
        #            "guarantor",
        #            "living_time",
        #            "assets",
        #            "age",
        #            "other_credits",
        #            "accomodation",
        #            "previous_credits",
        #            "job",
        #            "persons",
        #            "telephone",
        #            "foreign_worker",
        #            "credit_risk"]
        # data.columns = columns
        return data

    def make_data_set(self, data):
        """
        This function makes the dataset.
        """
        data.drop(['status'],axis = 1)
        X, y = data.drop('status', axis=1), data['status'].values
        y = y.reshape(len(y), 1)
        scaler = preprocessing.StandardScaler()
        X = scaler.fit_transform(X)
        dataset = np.append(X, y, axis=1).tolist()
        return dataset

    def euclides_distace(self, codebook1, codebook2):
        """
        This function calculates the euclides distance.
        """
        distance = 0.0
        for i in range(len(codebook1) - 1):
            distance += (codebook1[i] - codebook2[i]) ** 2
        return sqrt(distance)


    def random_codebook(self, dataset):
        """
        This function makes random codebook.
        """
        n_features = len(dataset[0])
        codebook = [dataset[randrange(len(dataset))][i] for i in range(n_features)]
        return codebook


    def best_matching_unit(self, codebook_list, data):
        """
        This function finds the best matching unit.
        """
        distances = list()
        for vector in codebook_list:
            distance = self.euclides_distace(vector, data)
            distances.append((vector, distance))
        distances.sort(key=lambda tup: tup[1])
        return distances[0][0]


    def LVQ_train(self, dataset, n_codebooks, learn_rate, total_epochs):
        codebooks = [self.random_codebook(dataset) for _ in range(n_codebooks)]
        x = [i for i in range(total_epochs)]
        y = []
        for epoch in range(total_epochs):
            print("------------------------------------------------------")
            learn_rate = learn_rate * (1.0 - (epoch / total_epochs))
            print(f'Epoch: {epoch}'
                  f'\nLearn rate: {learn_rate}')
            sse = 0
            for data in dataset:
                bmu = self.best_matching_unit(codebooks, data)
                for i in range(len(data)):
                    error =data[i]- bmu[i]
                    sse+= pow(error, 2)
                    if bmu[-1] == data[-1]:
                        bmu[i] += learn_rate * error
                    else:
                        bmu[i] -= learn_rate * error
            y.append(sse)

            print(f'SSE: {sse}')
        return codebooks