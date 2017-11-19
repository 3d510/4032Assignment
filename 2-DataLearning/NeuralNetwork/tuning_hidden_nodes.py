from __future__ import print_function
import matplotlib.pyplot as plt
import os, sys

from data_input import prepare_data, divide_into_folds
from network_train import nn_learning

#### Use 5-fold validation to decide best model - find optimal alpha

csv_file = 'movie_metadata_processed.csv'
if sys.argv >= 2:
    csv_file = sys.argv[1]
trainX, trainY, testX, testY = prepare_data(os.path.join('InputData', csv_file))
no_folds = 5
dataset = divide_into_folds(trainX, trainY, no_folds=no_folds)

input_layer_size = trainX.shape[1]
hidden_layer_size_s = [30, 60 ,90]
layers_size_s = [[input_layer_size, hidden_layer_size, 1] for hidden_layer_size in hidden_layer_size_s]

learning_rate = 1e-5
batch_size = 32
epochs = 1000


min_error = 1e+15
plot_style = ['-', '--', '-.', ':', ',']
plt.figure()
plt.suptitle('Train and validation error')
for j in range(len(layers_size_s)):
    layers_size = layers_size_s[j]
    hidden_layer_size = layers_size[1]
    print('*** Start examining hidden layer size:', hidden_layer_size)
    folds_error = 0.0
    plt.subplot(2, 3, j + 1)
    for i in range(len(dataset)):  # for each fold
        print('** Starting fold ', i + 1)
        train_X = dataset[i]['trainX']
        train_Y = dataset[i]['trainY']
        test_X = dataset[i]['testX']
        test_Y = dataset[i]['testY']
        train_results = nn_learning(train_X, train_Y, test_X, test_Y, learning_rate, layers_size, batch_size, epochs)
        folds_error += train_results['best_cost']

        plt.plot(train_results['train_cost'], 'r' + plot_style[i], label='Fold %d train error' % (i+1))
        plt.plot(train_results['test_cost'], 'b' + plot_style[i], label='Fold %d validation error' % (i+1))

    plt.xlabel('epoch')
    plt.ylabel('mean square error')
    if j == len(hidden_layer_size_s) - 1:
        plt.legend(loc='center left', bbox_to_anchor=(1.5, 0.5))
    plt.title('hidden layer=%d' % hidden_layer_size)

    mean_folds_error = folds_error / no_folds
    print('* Mean folds error:', mean_folds_error)
    if mean_folds_error < min_error:
        best_hidden_layer_size = hidden_layer_size
        best_layers_size = layers_size
        min_error = mean_folds_error

print ('Optimal hidden layer size:', best_hidden_layer_size)

plt.tight_layout()
plt.savefig(os.path.join('OutputData', 'hidden_node_tuning_train_validation_error.png'))
plt.show()

#### retrain model with best_learning_rate
train_results = nn_learning(trainX, trainY, testX, testY, learning_rate, best_layers_size, batch_size, epochs)
train_cost = train_results['train_cost']
test_cost = train_results['test_cost']

plt.figure()
plt.plot(test_cost, label='test cost')
plt.xlabel('epoch')
plt.ylabel('mean square error')
plt.legend()
plt.title('Test errors at optimal hidden size=%d' % best_hidden_layer_size)
plt.savefig(os.path.join('OutputData', 'hidden_node_tuning_test_error_optimal_alpha.png'))
plt.show()
