from LVQ import *

def main():
    neural_network = LVQ()
    data = neural_network.load_data('D:/desktop/Programming/Python/AI_ML/Neural networks/University Project/LVQ_NN/german_credit_data.csv')
    print(data.head())

if __name__ == '__main__':
    main()