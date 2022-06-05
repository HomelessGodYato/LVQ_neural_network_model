from random import shuffle, uniform
from typing import List, Union, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


class LVQ:
    """Implementacja algorytmu LVQ.

    Pozwala stwarzać, nauczać i używać wektorów kodujących (Codebook). Codebook jest listą wektorów.
    Wektor jest listą wag/atrybutów i klas.
    """

    """
    Konstruktor klasy LVQ
    """
    def __init__(
            self,
            codebook_size: int, # liczba wektorów kodujących
            features_count: int, # liczba atrybutów
            labels_count: int, # liczba klas
            codebook_init_method: str = "random", # metoda inicjalizacji codebook`a
            codebook_init_dataset: List[float] = None, # dane do inicjalizacji codebook`a
    ):

        self.codebook_size = codebook_size
        self.features_count = features_count
        self.labels_count = labels_count

        assert codebook_init_method in (
            "random",
        ), "Currently supported codebook initialization methods are: zeros, sample, random"
        if codebook_init_method == "sample":
            assert (
                    codebook_init_dataset is not None
            ), "Dataset is needed for sample codebook initialization"
            assert (
                    len(codebook_init_dataset) >= codebook_size
            ), "Not enough samples in the dataset"

        if codebook_init_method == "zeros":
            self._init_codebook_zeros()
        elif codebook_init_method == "sample":
            self._init_codebook_sample(codebook_init_dataset)
        elif codebook_init_method == "random":
            self._init_codebook_random()

    def _init_codebook_zeros(self) -> None:
        """Initialize codebook with zeros for all the features.

        Tries to take the same amout of samples for each label.
        """
        self.codebook = [
            [0] * self.features_count + [i % self.labels_count]
            for i in range(self.codebook_size)
        ]

    def _init_codebook_sample(self, dataset: List[List[float]]) -> None:
        """Initialize codebook based on sample dataset.

        Takes some samples from the dataset to initialize the codebook.
        Tries to take the same amout of samples for each label.
        """
        label_split = {label: [] for label in range(self.labels_count)}
        for vector in dataset:
            label_split[vector[-1]].append(vector)

        self.codebook = []
        idx = 0
        while len(self.codebook) < self.codebook_size:
            if len(label_split[idx]) > 0:
                self.codebook.append(label_split[idx].pop().copy())
            idx = (idx + 1) % self.labels_count

    def _init_codebook_random(self) -> None:
        """Initialize the codebook with random values bewteen 0 and 1.

        Tries to take the same amout of samples for each label.
        """
        self.codebook = [
            [uniform(0, 1) for _ in range(self.features_count)]
            + [i % self.labels_count]
            for i in range(self.codebook_size)
        ]

    @staticmethod
    def vector_euclidean_distance(a: List[float], b: List[float]) -> float:
        """Calculate Euclidean distance for all the features (ignore label)."""

        *features_a, _ = a
        *features_b, _ = b
        return sum([(v - x) ** 2 for v, x in zip(features_a, features_b)]) ** 0.5

    def get_best_matching_vector(self, input_vector: List[float]) -> List[float]:
        distances = []
        for vector in self.codebook:
            distances.append(self.vector_euclidean_distance(vector, input_vector))

        closest_vector = self.codebook[distances.index(min(distances))]
        return closest_vector

    def predict(self, input_features: List[float]) -> int:
        return self.get_best_matching_vector(input_features + [None])[-1]

    def update(
            self, train_vector: List[float], learning_rate: float
    ) -> Tuple[float, float]:
        best_vector = self.get_best_matching_vector(train_vector)
        error = 0.0
        for idx in range(self.features_count):
            error = train_vector[idx] - best_vector[idx]

            if train_vector[-1] == best_vector[-1]:
                best_vector[idx] += learning_rate * error
            else:
                best_vector[idx] -= learning_rate * error

        return best_vector[-1], error ** 2

    def train_codebook(
            self,
            train_vectors: List[List[float]],
            epochs: int,
            base_learning_rate: float,
            learning_rate_decay: Union[str, None] = "linear", ) -> List[float]:

        assert learning_rate_decay in [
            None,
            "linear",
        ], "Unsupported learning rate decay"

        progress = tqdm(
            range(epochs),
            unit="epochs",
            ncols=100,
            bar_format="Training: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt}{postfix}",
        )

        learning_rate = base_learning_rate
        y = [0]
        for epoch in progress:
            accuracy = 0
            if learning_rate_decay == "linear":
                learning_rate = self.linear_decay(base_learning_rate, epoch, epochs)
            for train_vector in train_vectors:
                prediction, square_error = self.update(train_vector, learning_rate)
                if prediction == train_vector[-1]:
                    accuracy += 1
            accuracy /= len(train_vectors)
            y.append(accuracy)
        y = [i * 100 for i in y]
        return y

    @staticmethod
    def linear_decay(base_rate: float, current_epoch: int, total_epochs: int) -> float:
        return base_rate * (1.0 - (current_epoch / total_epochs))


def cross_validate(
        # Validation params
        dataset: List[List[float]], fold_count: int,
        learning_rate: float, learning_rate_decay: Union[str, None], epochs: int,
        # Codebook params
        codebook_size: int, features_count: int,
        labels_count: int, codebook_init_method: str = "random",
        codebook_init_dataset: List[float] = None, model: LVQ = None):
    dataset_copy = dataset.copy()

    shuffle(dataset_copy)

    fold_size = len(dataset) // fold_count
    folds = [
        dataset_copy[idx: idx + fold_size] for idx in range(0, len(dataset), fold_size)
    ]

    scores = []
    accuracy_list = []
    iter = len(folds)
    i = 0
    folds_dict = {f'Fold {i}': '' for i in range(iter)}
    cf_list = dict.fromkeys(folds_dict.keys())
    print(folds_dict)

    for test_vectors in folds:
        label_list = []
        predictions_list = []
        print(f'Fold {len(scores) + 1}')
        train_vectors = folds.copy()
        train_vectors.remove(test_vectors)
        train_vectors = [item for fold in train_vectors for item in fold]
        correct = 0
        for vector in test_vectors:
            *features, label = vector
            label_list.append(label)
            prediction = model.predict(features)
            predictions_list.append(prediction)

            print(f'Predicted {prediction}: Expected {label}')
            if prediction == label:
                correct += 1
        cf_list[f'Fold {i}'] = confusion_matrix(y_true=label_list, y_pred=predictions_list)
        i += 1
        scores.append(correct / len(test_vectors))

    scores = [i * 100 for i in scores]

    return scores, cf_list, iter, accuracy_list


def make_plots(**kwargs):
    iter = kwargs['iter']
    epochs = kwargs['epochs']
    if 'scores' in kwargs:
        scores = kwargs['scores']
        fig = plt.figure(figsize=(12, 12))
        bars = sns.barplot(x=[i + 1 for i in range(len(scores))], y=scores)
        for bar in bars.patches:
            bars.annotate(f'{round(bar.get_height(), 2)}%',
                          (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                          ha='center', va='center',
                          xytext=(0, 10),
                          textcoords='offset points')
        plt.xlabel('Fold')
        plt.ylabel('Accuracy %')

    if 'confusion_matrixes' in kwargs:
        confusion_matrixes = kwargs['confusion_matrixes']
        fig, axes = plt.subplots(nrows=1, ncols=iter, figsize=(20, 4))
        fig.supxlabel('Actual')
        fig.supylabel('Predicted')
        folds_dict = {f'Fold {i}': '' for i in range(iter)}
        size = len(confusion_matrixes)
        for i, ax in enumerate(axes.flat):
            k = list(confusion_matrixes)[i]
            sns.heatmap(confusion_matrixes[k] / np.sum(confusion_matrixes[k]), ax=ax, annot=True, fmt='.2%',
                        cbar=True)
            ax.set_title(k, fontsize=8)

    if 'accuracy' in kwargs:
        accuracy = kwargs['accuracy']
        x = [i for i in range(epochs + 1)]
        y = accuracy
        plt.figure()
        plt.plot(x, y)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy %')
        plt.ylim(0, 100)
        plt.xlim(0, epochs)
        plt.xticks(x[::epochs // 10])
        plt.yticks(np.arange(0, 101, 5))
        plt.grid()
    plt.show()
