import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from random import randrange
from math import sqrt


df = pd.read_csv('D:/desktop/Programming/Python/AI_ML/Neural networks/University Project/LVQ_NN/german_credit_data.csv')
# plt.figure(figsize=(15,9))
# sns.heatmap(df.corr(), cmap='Blues', annot=True)
# plt.show()
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
df.columns = columns
df = df.drop(['sex/family'], axis=1)

X, y = df.drop('credit_risk', axis=1), df['credit_risk'].values
y = y.reshape(len(y), 1)
scaler = StandardScaler()
X = scaler.fit_transform(X)

dataset = np.append(X, y, axis=1).tolist()


def kfold(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)

    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)

    return dataset_split

n_folds = 5

folds = kfold(dataset, n_folds)

def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)-1):
        distance += (row1[i] - row2[i])**2
    return sqrt(distance)

def random_subset(train):
    n_records = len(train)
    n_features = len(train[0])
    subsets = [train[randrange(n_records)][i] for i in range(n_features)]
    return subsets


def best_match(subsets, test_row):
    distances = list()

    for subset in subsets:
        dist = euclidean_distance(subset, test_row)
        distances.append((subset, dist))

    distances.sort(key=lambda tup: tup[1])
    return distances[0][0]


def lvq(train_set, n_subsets, lrate, epochs):
    subsets = [random_subset(train_set) for i in range(n_subsets)]

    for epoch in range(epochs):
        print(epoch)
        rate = lrate * (1.0 - (epoch / float(epochs)))

        for row in train_set:
            bmu = best_match(subsets, row)

            for i in range(len(row) - 1):
                error = row[i] - bmu[i]
                if bmu[-1] == row[-1]:
                    bmu[i] += rate * error
                else:
                    bmu[i] -= rate * error

    return subsets

def accuracy(actual, predicted):
    correct = 0
    accuracy_list = []
    x_axis = []
    fig, ax = plt.subplots()
    ax.set_xlabel('epochs')
    ax.set_ylabel('accuracy')
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
            accuracy_list.append(correct / float(len(actual)) * 100.0)
            x_axis.append(i)
            ax.plot(x_axis, accuracy_list, color='red')
    return correct / float(len(actual)) * 100.0


def train_test_split(folds, fold):
    train_set = list(folds)
    train_set.remove(fold)
    train_set = sum(train_set, [])
    test_set = list()
    return train_set, test_set

lrate = 0.01
epochs = 20
n_subsets = 20

scores = list()

for fold in folds:
    train_set, test_set = train_test_split(folds, fold)

    for row in fold:
        test_set.append(list(row))

    subsets = lvq(train_set, n_subsets, lrate, epochs)
    y_hat = list()

    for test_row in test_set:
        output = best_match(subsets, test_row)[-1]
        y_hat.append(output)

    y = [row[-1] for row in fold]
    scores.append(accuracy(y, y_hat))
    plt.show()


print(f'Accuracy per fold: {scores}')
print(f'Max Accuracy: {scores}')

