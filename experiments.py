import random

import data
import LVQ
from LVQ import make_plots

"""
This script implements some experiments to test LVQ model
"""

# printing vectors
def pprint_vector(vector):
    vector = [str(round(value, 3)) for value in vector]
    if len(vector) > 6:
        vector = [*vector[:3], "...", *vector[-3:]]
    print("[" + ", ".join([f"{value:>7}" for value in vector]) + "]")


# main function for experiments
def main_algorithm(labels_mapping,
                   dataset,
                   epochs,
                   learning_rate,
                   codebooks_count,
                   folds):

    # printing all unique output classes
    print("Label mapping:")
    print(labels_mapping)

    # randomly assigned codebook vector
    sample = random.choice(dataset)

    # calculating length of codebook vector, number of features
    *features, label = sample
    features_count = len(features)
    labels_count = len(labels_mapping)

    # number of codebook vectors that will used to train LVQ model
    codebooks_count = codebooks_count

    # LVQ neural network initialization
    model = lvq_test.LVQ(codebooks_count,
                         features_count,
                         labels_count,
                         "sample",
                         dataset)

    # assigned codebook vectors (initial values)
    print("Initialized codebook:")
    for vector in model.codebook:
        pprint_vector(vector)

    # LVQ model trainig
    print("Training model...")
    print("Innitial model:")
    accuracy_list, accuracy = model.train_codebook(
        train_vectors=dataset,
        base_learning_rate=learning_rate,
        learning_rate_decay=None,
        epochs=epochs,
    )

    # wypisanie nauczonych codebooków
    print("Trained codebook:")
    for vector in model.codebook:
        pprint_vector(vector)

    # incjalizacja cross-walidacji
    print("Cross validating model...")
    scores, confusion_matrixes, iter = lvq_test.cross_validate(
        dataset,
        folds,
        learning_rate=0.01,
        learning_rate_decay=None,
        epochs=epochs,
        codebooks_count=codebooks_count,
        features_count=features_count,
        labels_count=labels_count,
        codebook_init_method="sample",
        model=model
    )
    # EXPERIMENT 1: how learning rate affects model accuracy

    # model1 = lvq_test.LVQ(codebooks_count,
    #                       features_count,
    #                       labels_count,
    #                       "sample",
    #                       dataset)
    # print("Learning rate experiment:")
    # learning_rate_list = [0.01, 0.05, 0.09, 0.13,
    #                       0.17, 0.21, 0.25, 0.29,
    #                       0.33, 0.37, 0.41, 0.45,
    #                       0.49, 0.53, 0.57, 0.61,
    #                       0.65, 0.69, 0.73, 0.77,
    #                       0.81, 0.85, 0.89, 0.93, 0.97]
    # results1 = []
    # parameters_learning_rate = {}
    # for i in learning_rate_list:
    #     accuracy_list, accuracy = model1.train_codebook(
    #         train_vectors=dataset,
    #         base_learning_rate=i,
    #         learning_rate_decay='linear',
    #         epochs=epochs,
    #     )
    #     parameters_learning_rate[i] = accuracy
    #     results1.append(accuracy)
    # max_accuracy = max(parameters_learning_rate.values())
    # best = [key for key, val in parameters_learning_rate.items() if val == max_accuracy]
    # best_lr = best[0]
    # print(len(learning_rate_list) == len(results1))
    #

    # EXPERIMENT 2: how number of epochs affects model accuracy
    # print("Epochs experiment:")
    # model2 = lvq_test.LVQ(codebooks_count,
    #                       features_count,
    #                       labels_count,
    #                       "sample",
    #                       dataset)
    # epochs_range = [10, 50, 100, 150, 200,
    #                 250, 300, 350, 400,
    #                 450, 500, 550, 600,
    #                 650, 700, 750, 800,
    #                 850, 900, 950, 1000]
    # results2 = []
    # parameters_epochs = {}
    # for i in epochs_range:
    #     accuracy_list, accuracy = model2.train_codebook(
    #         train_vectors=dataset,
    #         base_learning_rate=0.01,
    #         learning_rate_decay='linear',
    #         epochs=i,
    #     )
    #     print(accuracy)
    #     parameters_epochs[i] = accuracy
    #     results2.append(accuracy)
    # max_accuracy = max(parameters_epochs.values())
    # best = [key for key, val in parameters_epochs.items() if val == max_accuracy]
    # best_epoch = best[0]


    # EXPERIMENT 3: how number of codebook vectors affects model accuracy
    # print("Codebook size experiment:")
    # codebooks_sizes = [1, 5, 10, 20, 30,
    #                    40, 50, 100, 200,
    #                    400, 600, 800, 900, 950]
    # results3 = []
    # parameters_codebooks_count = {}
    #
    # for i in codebooks_sizes:
    #     model3 = lvq_test.LVQ(i,
    #                           features_count,
    #                           labels_count,
    #                           "sample",
    #                           dataset)
    #     accuracy_list, accuracy = model3.train_codebook(
    #         train_vectors=dataset,
    #         base_learning_rate=0.01,
    #         learning_rate_decay='linear',
    #         epochs=100,
    #     )
    #     parameters_codebooks_count[i] = accuracy
    #     results3.append(accuracy)
    # max_accuracy = max(parameters_codebooks_count.values())
    # best = [key for key, val in parameters_codebooks_count.items() if val == max_accuracy]
    # best_codebooks_count = best[0]

    # EXPERIMENT 4: trainig LVQ model with the best parameters that are found in previous experiments
    # print("Best parameters experiment:")
    # model3 = lvq_test.LVQ(best_codebooks_count,
    #                       features_count,
    #                       labels_count,
    #                       "sample",
    #                       dataset)
    # best_accuracy_list, best_accuracy = model3.train_codebook(train_vectors=dataset,
    #                                                           base_learning_rate=best_lr,
    #                                                           learning_rate_decay='linear',
    #                                                           epochs=best_epoch, )
    #epoch = 250
    #learning_rate = 0.01
    #learning_rate = 0.13


    # give parameters you need and this function will make plots for you
    make_plots(epochs=epochs,
               iter=iter,
               scores=scores,
               confusion_matrixes=confusion_matrixes,
               accuracy=accuracy_list
               )

    return accuracy, accuracy_list

# function that starts experiment
def experiments():
    print("Starting...")
    print("---------------------------------------------------------")
    # use path for your dataset
    labels_mapping, dataset = data.load_data_normalization(
        'D:/desktop/Programming/Python/AI_ML/Neural networks/University Project/LVQ_NN/german_credit_data.csv')
    main_algorithm(labels_mapping, dataset, 100, 0.01, 30, 5)


experiments()
