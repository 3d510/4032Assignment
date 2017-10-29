from __future__ import print_function
import matplotlib.pyplot as plt
import os

from data_input import prepare_data, divide_into_folds
from network_train import nn_learning

#### Use 5-fold validation to decide best model - find optimal alpha

trainX, trainY, testX, testY = prepare_data()

no_folds = 5
dataset = divide_into_folds(trainX, trainY, no_folds=no_folds)

input_layer_size = trainX.shape[1]
hidden_layer_size = 30
layers_size = [input_layer_size, hidden_layer_size, 1]

learning_rates = [1e-3, 0.5e-3, 1e-4, 0.5e-4, 1e-5]
batch_size = 1
epochs = 1000


min_error = 1e+18
plot_style = ['-', '--', '-.', ':', ',']
plt.figure(figsize=(8.0, 5.0))
plt.suptitle('Train and validation error')
for j in range(len(learning_rates)):
    learning_rate = learning_rates[j]
    print('*** Start examining learning rate:', learning_rate)
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
    if j == len(learning_rates) - 1:
        plt.legend(loc='center left', bbox_to_anchor=(1.5, 0.5))
    plt.title('alpha=%.5f' % learning_rate)

    mean_folds_error = folds_error / no_folds
    print ('* Mean folds error:', mean_folds_error)
    if mean_folds_error < min_error:
        best_learning_rate = learning_rate
        min_error = mean_folds_error

print('Optimal learning rate:', best_learning_rate)

plt.tight_layout()
plt.savefig(os.path.join('OutputData', 'q2_train_validation_error_GD.png'))
plt.show()

#### retrain model with best_learning_rate
train_results = nn_learning(trainX, trainY, testX, testY, best_learning_rate, layers_size, batch_size, epochs)
train_cost = train_results['train_cost']
test_cost = train_results['test_cost']

plt.figure(figsize=(8.0, 5.0))
plt.plot(test_cost)
plt.xlabel('epoch')
plt.ylabel('mean square error')
plt.title('Test errors at optimal alpha=%.5f' % best_learning_rate)
plt.savefig(os.path.join('OutputData', 'q2_test_error_optimal_alpha.png'))
plt.show()
