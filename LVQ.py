import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, datasets, linear_model
from sklearn.model_selection import train_test_split,KFold
import seaborn as sns
import random
from random import seed

class LVQ:

    def load_data(self,file_name):
        """
        This function loads the dataset.
        """
        data = pd.read_csv(file_name)
        return data

    def change_columns_name(self,data):
        columns = ["current_account",
                   "duration",
                   "behavior",
                   "purpose",
                   "amount",
                   "savings",
                   "employment_time",
                   "installment_rate",
                   "sex/family",
                   "guarantor",
                   "living_time",
                   "assets",
                   "age",
                   "other_credits",
                   "accomodation",
                   "previous_credits",
                   "job",
                   "persons",
                   "telephone",
                   "foreign_worker",
                   "credit_risk"
                   ]
        data.columns = columns
        return data
    def prepare_data(self,data):
        data = data.drop(['purpose'], axis=1)
        return data

    def normalize(self,data):
        data = preprocessing.normalize(data)
        return data

    def euclides_distace(self,row1,row2):
        distance = 0
        for i in range(len(row1)-1):
            distance += (row1[i] - row2[i])**2
        return np.sqrt(distance)

    def best_matching_unit(self,codebooks,row):
        distances = []
        for codebook in codebooks:
            dist = self.euclides_distace(codebook,row)
            distances.append((codebook,dist))
        distances.sort(key=lambda tup: tup[1])
        return distances[0][0]

    def random_codebook(self,train):
        n_records = len(train)
        n_features = len(train[0])
        codebook = [train[random.randrange(n_records)][i] for i in range(n_features)]
        return codebook

    def train_codebooks(self,train, n_codebooks, lrate, epochs):
        codebooks = [self.random_codebook(train) for i in range(n_codebooks)]
        for epoch in range(epochs):
            rate = lrate * (1.0 - (epoch / float(epochs)))
            sum_error = 0.0
            for row in train:
                bmu = self.best_matching_unit(codebooks, row)
                for i in range(len(row) - 1):
                    error = row[i] - bmu[i]
                    sum_error += error ** 2
                    if bmu[-1] == row[-1]:
                        bmu[i] += rate * error
                    else:
                        bmu[i] -= rate * error
            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, rate, sum_error))
        return codebooks

    def cross_validation_split(self,dataset, n_folds):
        dataset_split = list()
        dataset_copy = list(dataset)
        fold_size = int(len(dataset) / n_folds)
        for i in range(n_folds):
            fold = list()
            while len(fold) < fold_size:
                index = random.randrange(len(dataset_copy))
                fold.append(dataset_copy.pop(index))
            dataset_split.append(fold)
        return dataset_split

    # Calculate accuracy percentage
    def accuracy_metric(self,actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0

    def evaluate_algorithm(self,dataset, algorithm, n_folds, *args):
        folds = self.cross_validation_split(dataset, n_folds)
        scores = list()
        for fold in folds:
            train_set = list(folds)
            train_set.remove(fold)
            train_set = sum(train_set, [])
            test_set = list()
            for row in fold:
                row_copy = list(row)
                test_set.append(row_copy)
                row_copy[-1] = None
            predicted = algorithm(train_set, test_set, *args)
            actual = [row[-1] for row in fold]
            accuracy = self.accuracy_metric(actual, predicted)
            scores.append(accuracy)
        return scores

    def predict(self,codebooks, test_row):
        bmu = self.best_matching_unit(codebooks, test_row)
        return bmu[-1]

    def learning_vector_quantization(self,train, test, n_codebooks, lrate, epochs):
        codebooks = self.train_codebooks(train, n_codebooks, lrate, epochs)
        predictions = list()
        for row in test:
            output = self.predict(codebooks, row)
            predictions.append(output)
        return (predictions)