from LVQ import *

def main():
    neural_network = LVQ()
    data = neural_network.load_data('D:/desktop/Programming/Python/AI_ML/Neural networks/University Project/LVQ_NN/german_credit_data.csv')
    data = neural_network.change_columns_name(data)
    data = neural_network.prepare_data(data)
    codebooks = np.array(data)
    row = codebooks[0]
    bmu = neural_network.best_matching_unit(codebooks,row)
    seed(1)
    n_folds = 6
    learn_rate = 0.3
    n_epochs = 50
    n_codebooks = 50
    scores = neural_network.evaluate_algorithm(codebooks, neural_network.learning_vector_quantization, n_folds, n_codebooks, learn_rate, n_epochs)
    print('Scores: %s' % scores)
    print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
    # learning_rate =0.001
    # epochs = 10
    # n_codebooks = 10
    # codebooks_train = neural_network.train_codebooks(codebooks, n_codebooks, learning_rate, epochs)
    # print('Codebooks: %s' % codebooks)


if __name__ == '__main__':
    main()