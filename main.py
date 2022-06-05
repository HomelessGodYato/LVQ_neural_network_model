import random
from data import *
from lvq_test import *


def pprint_vector(vector):
    vector = [str(round(value, 3)) for value in vector]
    if len(vector) > 6:
        vector = [*vector[:3], "...", *vector[-3:]]
    print("[" + ", ".join([f"{value:>7}" for value in vector]) + "]")


if __name__ == "__main__":
    labels_mapping, dataset = load_data('D:/desktop/Programming/Python/AI_ML/Neural networks/University Project/LVQ_NN/sorted_credit.csv')

    print("Label mapping:")
    print(labels_mapping)
    print(dataset)
    sample = random.choice(dataset)
    *features, label = sample

    features_count = len(features)
    labels_count = len(labels_mapping)

    model = LVQ(10, features_count, labels_count, "random", dataset)
    print("Random sample:")
    pprint_vector(sample)

    print("Prediction:", model.predict(features))
    print("Initialized codebook:")
    for vector in model.codebook:
        pprint_vector(vector)

    print("Training model...")
    accuracy =  model.train_codebook(
        train_vectors=dataset,
        base_learning_rate=0.01,
        learning_rate_decay=None,
        epochs=100,
    )

    print("Prediction:", model.predict(features))
    print("Trained codebook:")
    for vector in model.codebook:
        pprint_vector(vector)

    print("Cross validating model...")
    scores, confusion_matrixes, iter, accuracy_list = cross_validate(
        dataset,
        5,
        learning_rate=0.01,
        learning_rate_decay=None,
        epochs=100,
        codebook_size=10,
        features_count=features_count,
        labels_count=labels_count,
        codebook_init_method="random",
        model = model
    )
    make_plots(scores=scores, confusion_matrixes=confusion_matrixes, iter=iter, epochs=100, accuracy = accuracy, accuracy_list=accuracy_list)

    print("Cross validation scores:", [round(score, 3) for score in scores])
    print("Average:", round(sum(scores) / len(scores), 3))


