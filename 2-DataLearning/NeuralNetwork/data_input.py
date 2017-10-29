import numpy as np
import pandas as pd
import os


def read_data(file_name):
    df = pd.read_csv(file_name)
    movie = df.values
    number_of_columns = len(df.columns)
    X_data, Y_data = movie[:, :number_of_columns-1], movie[:, -1]
    Y_data = (np.asmatrix(Y_data)).transpose()
    return X_data, Y_data


def scale(X, X_min, X_max):
    return (X - X_min) / (X_max - X_min)


def shuffle_data(samples, labels):
    idx = np.arange(samples.shape[0])
    np.random.shuffle(idx)
    # print  (samples.shape, labels.shape)
    samples, labels = samples[idx], labels[idx]
    return samples, labels


def prepare_data(filename):
    X_data, Y_data = read_data(filename)
    X_data, Y_data = shuffle_data(X_data, Y_data)
    no_examples = X_data.shape[0]
    # separate train and test data
    m = 3 * no_examples // 10
    testX, testY = X_data[:m], Y_data[:m]
    trainX, trainY = X_data[m:], Y_data[m:]

    # scale data
    trainX_max, trainX_min = np.max(trainX[:, :8], axis=0), np.min(trainX[:, :8], axis=0)
    testX_max, testX_min = np.max(testX[:, :8], axis=0), np.min(testX[:, :8], axis=0)

    trainX[:, :8] = scale(trainX[:, :8], trainX_min, trainX_max)
    testX[:, :8] = scale(testX[:, :8], testX_min, testX_max)

    return trainX, trainY, testX, testY


def divide_into_folds(dataX, dataY, no_folds):
    dataX, dataY = shuffle_data(dataX, dataY)
    no_examples = dataX.shape[0]
    partition_size = no_examples // no_folds
    dataset = []
    for i in range(no_folds):
        start, end = i * partition_size, (i+1) * partition_size
        if i == no_folds - 1:
            end = no_examples
        dataset.append({
            'trainX': np.append(dataX[:start], dataX[end:], axis=0),
            'trainY': np.append(dataY[:start], dataY[end:], axis=0),
            'testX': dataX[start:end],
            'testY': dataY[start:end]
        })
    return dataset
