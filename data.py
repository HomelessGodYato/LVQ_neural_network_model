import random
from typing import List

import pandas as pd


# wczytywanie danych bez normalizacji
def load_data(filename):
    dataset = pd.read_csv(filename, header=None)
    dataset = dataset.values
    labels = dataset[:, -1]
    mapping = set(labels)
    return mapping, dataset


# wczytywanie danych  i normalizacja
def load_data_normalization(filename):
    dataset = pd.read_csv(filename, header=None)
    dataset = dataset.values
    labels = dataset[:, -1]
    mapping = set(labels)

    normalized = normalize(dataset)
    normalized = random.sample(normalized, len(normalized))
    return mapping, normalized


# funkcja normalizacji
def normalize(dataset: List[List[float]]) -> List[List[float]]:
    feature_count = len(dataset[0]) - 1

    total_min = [None] * feature_count
    total_max = [None] * feature_count
    for *features, _ in dataset:
        for idx, feature in enumerate(features):
            if total_min[idx] is None or total_min[idx] > feature:
                total_min[idx] = feature
            if total_max[idx] is None or total_max[idx] < feature:
                total_max[idx] = feature

    normalized_dataset = []
    for *features, label in dataset:
        normalized_dataset.append(
            [
                (feature - min_) / (max_ - min_)
                for feature, min_, max_ in zip(features,
                                               total_min,
                                               total_max)
            ]
            + [label]
        )
    return normalized_dataset
