from random import shuffle
from typing import List, Union, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


class LVQ:
    def __init__(
            self,
            codebooks_count: int,  # numbers of codebooks
            features_count: int,  # features number
            labels_count: int,  # class (output) number
            codebook_init_method: str = "random",  # codebook initialization method
            codebook_init_dataset: List[float] = None,  # dataset for initialization codebooks
    ):

        self.codebooks_count = codebooks_count
        self.features_count = features_count
        self.labels_count = labels_count

        # algorithm takes random samples from dataset
        # TODO: implement more methods (zeros, random, etc.)

        assert codebook_init_method in (
            "sample",
        )

        if codebook_init_method == "sample":
            assert (
                    codebook_init_dataset is not None
            ), "Dataset is needed for sample codebook initialization"
            assert (
                    len(codebook_init_dataset) >= codebooks_count
            ), "Not enough samples in the dataset"

        if codebook_init_method == "sample":
            self._init_codebook_sample(codebook_init_dataset)

    # dividing dataset to output class subsets
    def _init_codebook_sample(self, dataset: List[List[float]]) -> None:
        label_split = {label: [] for label in range(self.labels_count)}
        for vector in dataset:
            label_split[vector[-1]].append(vector)

        self.codebook = []
        idx = 0
        # assigning data to codebooks
        while len(self.codebook) < self.codebooks_count:
            if len(label_split[idx]) > 0:
                self.codebook.append(label_split[idx].pop().copy())
            idx = (idx + 1) % self.labels_count

    @staticmethod
    # Euclidean distance calculation
    def vector_euclidean_distance(a: List[float], b: List[float]) -> float:
        *features_a, _ = a
        *features_b, _ = b
        return sum([(v - x) ** 2 for v, x in zip(features_a, features_b)]) ** 0.5

    # finding best matching unit (BMU)
    def get_best_matching_vector(self, input_vector: List[float]) -> List[float]:
        distances = []
        for vector in self.codebook:
            distances.append(self.vector_euclidean_distance(vector, input_vector))

        closest_vector = self.codebook[distances.index(min(distances))]
        return closest_vector

    # prediction for particular codebook or given vector
    def predict(self, input_features: List[float]) -> int:
        return self.get_best_matching_vector(input_features + [None])[-1]

    # updating BMU position (updating weight matrix)
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

        # LVQ model training

    def train_codebook(
            self,
            train_vectors: List[List[float]],
            epochs: int,
            base_learning_rate: float,
            learning_rate_decay: Union[str, None] = "linear", ) -> List[float]:
        assert learning_rate_decay in [
            None,
            "linear",
        ]
        # just for pretty output in terminal
        progress = tqdm(
            range(epochs),
            unit="epochs",
            ncols=100,
            bar_format="Training: {percentage:3.0f}% |{bar}| {n_fmt}/{total_fmt}{postfix}",
        )

        # base (start) learning rate initialization
        learning_rate = base_learning_rate
        y = [0]  # accuracy list initialization
        for epoch in progress:
            accuracy = 0
            if learning_rate_decay == "linear":
                # learning rate decay for each epoch
                learning_rate = self.linear_decay(base_learning_rate, epoch, epochs)
            for train_vector in train_vectors:
                # updating codebook vectors
                prediction, square_error = self.update(train_vector, learning_rate)
                if prediction == train_vector[-1]:
                    accuracy += 1  # incrementing accuracy
            accuracy /= len(train_vectors)
            y.append(accuracy)
        y = [i * 100 for i in y]  # converting accuracy list to percents
        return y, accuracy * 100  # returning data for making plots in future

    @staticmethod
    def linear_decay(base_rate: float, current_epoch: int, total_epochs: int) -> float:
        return base_rate * (1.0 - (current_epoch / total_epochs))
    # TODO: implement more learning rate decay methods (discrete, exponential, etc.)


def cross_validate(
        # Validation params
        dataset: List[List[float]], fold_count: int,
        learning_rate: float, learning_rate_decay: Union[str, None], epochs: int,
        # Codebook params
        codebook_size: int, features_count: int,
        labels_count: int, codebook_init_method: str = "sample",
        codebook_init_dataset: List[float] = None, model: LVQ = None):
    # dividing dataset to subsets (folds)
    dataset_copy = dataset.copy()
    shuffle(dataset_copy)

    # calculating subsets (folds) size
    fold_size = len(dataset) // fold_count
    folds = [
        dataset_copy[idx: idx + fold_size] for idx in range(0, len(dataset), fold_size)
    ]

    scores = []  # cross-validation scores list
    iter = len(folds)
    i = 0
    folds_dict = {f'Fold {i + 0}': '' for i in range(iter)}
    cf_list = dict.fromkeys(folds_dict.keys())
    print(folds_dict)

    for test_vectors in folds:
        #
        label_list = []
        predictions_list = []
        print(f'Fold {len(scores) + 1}')
        train_vectors = folds.copy()
        train_vectors.remove(test_vectors)
        train_vectors = [item for fold in train_vectors for item in fold]
        correct = 0  # number of correctly guessed classes (outputs)
        for vector in test_vectors:
            *features, label = vector
            label_list.append(label)
            prediction = model.predict(features)
            predictions_list.append(prediction)

            print(f'Predicted {prediction}: Expected {label}')
            if prediction == label:
                correct += 1
        cf_list[f'Fold {i}'] = confusion_matrix(y_true=label_list,
                                                y_pred=predictions_list)  # confusion matrix
        i += 1
        scores.append(correct / len(test_vectors))

    scores = [i * 100 for i in scores]  # converting scores to percents

    return scores, cf_list, iter  # returning data for making plots in future


# making plots
def make_plots(**kwargs):
    # for cross-validation plot
    if 'iter' in kwargs:
        iter = kwargs['iter']

    # for most of the plots
    if 'epochs' in kwargs:
        epochs = kwargs['epochs']

    # just plotting
    if 'scores' in kwargs:
        scores = kwargs['scores']
        fig = plt.figure(figsize=(12, 12))
        plt.title('Cross-validation results (bar plot).')
        bars = sns.barplot(x=[i + 1 for i in range(len(scores))], y=scores)
        for bar in bars.patches:
            bars.annotate(f'{round(bar.get_height(), 2)}%',
                          (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                          ha='center', va='center',
                          xytext=(0, 10),
                          textcoords='offset points')
        plt.xlabel('Fold')
        plt.ylabel('Accuracy %')

    if 'confusion_matrices' in kwargs:
        confusion_matrices = kwargs['confusion_matrices']
        fig, axes = plt.subplots(nrows=1, ncols=iter, figsize=(20, 4))
        fig.supxlabel('Actual')
        fig.supylabel('Predicted')
        fig.suptitle('Cross-validation results (confusion matrices).', fontsize=16)
        folds_dict = {f'Fold {i}': '' for i in range(iter)}
        size = len(confusion_matrices)
        for i, ax in enumerate(axes.flat):
            k = list(confusion_matrices)[i]
            sns.heatmap(confusion_matrices[k] / np.sum(confusion_matrices[k]), ax=ax, annot=True, fmt='.2%',
                        cbar=True)
            ax.set_title(k, fontsize=8)

    if 'accuracy' in kwargs:
        accuracy = kwargs['accuracy']
        x = [i for i in range(epochs + 1)]
        y = accuracy
        plt.figure()
        plt.plot(x, y)
        plt.title('Neural network accuracy plot.')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy %')
        plt.ylim(0, 100)
        plt.xlim(0, epochs)
        plt.xticks(x[::epochs // 10])
        plt.yticks(np.arange(0, 101, 5))
        plt.grid()

    if 'learning_rate' in kwargs:
        learning_rate = kwargs['learning_rate']
        x = learning_rate
        y = kwargs['results1']
        plt.figure(figsize=(12, 20))
        plt.plot(x, y)
        plt.title('Learning rate impact on network accuracy')
        plt.xlabel('Learning rate')
        plt.ylabel('Accuracy %')
        plt.ylim(min(y) - 10, 100)
        plt.xlim(0, learning_rate[-1])
        plt.xticks(learning_rate)
        plt.grid(axis='y')

    if 'epochs_range' in kwargs:
        epochs_range = kwargs['epochs_range']
        x = epochs_range
        y = kwargs['results2']
        plt.figure(figsize=(12, 20))
        plt.plot(x, y)
        plt.title('Epochs number impact on network accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy %')
        plt.ylim(min(y) - 5, max(y) + 5)
        plt.xlim(0, epochs_range[-1])
        plt.xticks(x)
        plt.grid(axis='y')

    if 'codebooks_count' in kwargs:
        codebooks_count = kwargs['codebooks_count']
        x = codebooks_count
        y = kwargs['results3']
        plt.figure(figsize=(12, 20))
        plt.plot(x, y)
        plt.title('Codebooks number impact on network accuracy')
        plt.xlabel('Codebook size')
        plt.ylabel('Accuracy %')
        plt.ylim(min(y) - 5, max(y) + 5)
        plt.xlim(0, codebooks_count[-1])
        plt.xticks(x)
        plt.grid(axis='y')

    if 'best_accuracy' in kwargs:
        best_accuracy = kwargs['best_accuracy']
        best_epochs = kwargs['best_epochs']
        x = [i for i in range(best_epochs + 1)]
        y = best_accuracy
        plt.figure()
        plt.plot(x, y)
        plt.title('Accuracy with the best parameters.')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy %')
        plt.ylim(0, 100)
        plt.xlim(0, best_epochs)
        plt.xticks(x[::best_epochs // 10])
        plt.yticks(np.arange(0, 101, 5))
        plt.grid()

    plt.show()
