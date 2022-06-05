import random

import data
import lvq_test
from lvq_test import cross_validate, make_plots


def pprint_vector(vector):
    vector = [str(round(value, 3)) for value in vector]
    if len(vector) > 6:
        vector = [*vector[:3], "...", *vector[-3:]]
    print("[" + ", ".join([f"{value:>7}" for value in vector]) + "]")


def main_algorithm(labels_mapping, dataset, epochs, learning_rate, folds):
    print("Label mapping:")
    print(labels_mapping)

    sample = random.choice(dataset)
    *features, label = sample

    features_count = len(features)
    labels_count = len(labels_mapping)
    codebook_size = 30

    model = lvq_test.LVQ(codebook_size, features_count, labels_count, "sample", dataset)
    print("Random sample:")
    pprint_vector(sample)

    print("Prediction:", model.predict(features))
    print("Initialized codebook:")
    for vector in model.codebook:
        pprint_vector(vector)

    print("Training model...")
    accuracy = model.train_codebook(
        train_vectors=dataset,
        base_learning_rate=learning_rate,
        learning_rate_decay=None,
        epochs=epochs,
    )

    print("Prediction:", model.predict(features))
    print("Trained codebook:")
    for vector in model.codebook:
        pprint_vector(vector)

    print("Cross validating model...")
    scores, confusion_matrixes, iter, accuracy_list = cross_validate(
        dataset,
        folds,
        learning_rate=0.01,
        learning_rate_decay=None,
        epochs=epochs,
        codebook_size=codebook_size,
        features_count=features_count,
        labels_count=labels_count,
        codebook_init_method="sample",
        model=model
    )
    make_plots(scores=scores, confusion_matrixes=confusion_matrixes, iter=iter, epochs=epochs, accuracy=accuracy,
               accuracy_list=accuracy_list)


def experiments():
    print("Raw data")
    print("---------------------------------------------------------")
    # labels_mapping, dataset = data.load_data_normalization(
    #     'D:/desktop/Programming/Python/AI_ML/Neural networks/University Project/LVQ_NN/credit.csv')
    # main_algorithm(labels_mapping, dataset, 100, 0.01, 5)
    #
    print("Prepared_data")
    print("---------------------------------------------------------")
    labels_mapping, dataset = data.load_data_normalization(
        'D:/desktop/Programming/Python/AI_ML/Neural networks/University Project/LVQ_NN/sorted_credit.csv')
    main_algorithm(labels_mapping, dataset, 100, 0.01, 5)
    #
    # print("Prepared_data_without_normalization")
    # print("---------------------------------------------------------")
    # labels_mapping, dataset = data.load_data(
    # 'D:/desktop/Programming/Python/AI_ML/Neural networks/University Project/LVQ_NN/sorted_credit.csv')
    # main_algorithm(labels_mapping, dataset, 100, 0.01, 2)

    # print("10 epochs")
    # print("---------------------------------------------------------")
    # labels_mapping, dataset = data.load_data_normalization(
    # 'D:/desktop/Programming/Python/AI_ML/Neural networks/University Project/LVQ_NN/sorted_credit.csv')
    # main_algorithm(labels_mapping, dataset, 10, 0.03, 5)
    #
    # print("100 epochs")
    # print("---------------------------------------------------------")
    # labels_mapping, dataset = data.load_data_normalization(
    # 'D:/desktop/Programming/Python/AI_ML/Neural networks/University Project/LVQ_NN/sorted_credit.csv')
    # main_algorithm(labels_mapping, dataset, 100, 0.03, 5)
    #
    # print("200 epochs")
    # print("---------------------------------------------------------")
    # labels_mapping, dataset = data.load_data_normalization(
    # 'D:/desktop/Programming/Python/AI_ML/Neural networks/University Project/LVQ_NN/sorted_credit.csv')
    # main_algorithm(labels_mapping, dataset, 200, 0.03, 5)
    #
    # print("500 epochs")
    # print("---------------------------------------------------------")
    # labels_mapping, dataset = data.load_data_normalization(
    # 'D:/desktop/Programming/Python/AI_ML/Neural networks/University Project/LVQ_NN/sorted_credit.csv')
    # main_algorithm(labels_mapping, dataset, 500, 0.01, 5)

    # print("1000 epochs")
    # print("---------------------------------------------------------")
    # labels_mapping, dataset = data.load_data_normalization(
    #     'D:/desktop/Programming/Python/AI_ML/Neural networks/University Project/LVQ_NN/sorted_credit.csv')
    # main_algorithm(labels_mapping, dataset, 1000, 0.01, 5)


experiments()
