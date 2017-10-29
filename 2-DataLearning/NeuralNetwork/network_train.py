from __future__ import print_function
import numpy as np
import theano
import theano.tensor as T
from data_input import shuffle_data

np.random.seed(10)
floatX = theano.config.floatX

def init_bias(n = 1):
    # return(theano.shared(np.zeros(n), theano.config.floatX))
    return theano.shared(np.random.randn(n)*0.01, theano.config.floatX)


def init_weights(n_in, n_out):
    # W_values = np.random.uniform(low=-np.sqrt(6. / (n_in + n_out)),
    #                              high=np.sqrt(6. / (n_in + n_out)),
    #                              size=(n_in, n_out))
    W_values = np.random.randn(n_in, n_out) * 0.01
    # if logistic:
    #     W_values *= 4
    return theano.shared(W_values, theano.config.floatX)


def nn_learning(trainX, trainY, testX, testY, learning_rate, layers_size, batch_size, epochs):

    x = T.matrix('x')
    d = T.matrix('d')

    # initialize weights and biases for hidden layer(s) and output layer
    w1 = init_weights(layers_size[0], layers_size[1])
    b1 = init_bias(layers_size[1])
    w2 = init_weights(layers_size[1], layers_size[2])
    b2 = init_bias(layers_size[2])

    # learning rate
    alpha = theano.shared(learning_rate, floatX)

    # define mathematical expression - forward propagation
    u1 = T.dot(x, w1) + b1
    y1 = T.nnet.sigmoid(u1)
    u2 = T.dot(y1, w2) + b2
    y2 = T.nnet.sigmoid(u2) * 10
    pred = y2

    cost = T.abs_(T.mean(T.sqr(d - pred)))
    accuracy = T.abs_(T.mean(d - pred))

    # define gradients - back propagation
    dw2, db2, dw1, db1 = T.grad(cost, [w2, b2, w1, b1])
    updates = [
        [w2, w2 - alpha * dw2],
        [b2, b2 - alpha * db2],
        [w1, w1 - alpha * dw1],
        [b1, b1 - alpha * db1]
    ]

    # define train and predict function

    train = theano.function(
        inputs=[x, d],
        outputs=cost,
        updates=updates,
        allow_input_downcast=True
    )

    test = theano.function(
        inputs=[x, d],
        outputs=[pred, cost, accuracy],
        allow_input_downcast=True
    )

    # start learning process
    train_cost = np.zeros(epochs)
    test_cost = np.zeros(epochs)
    test_accuracy = np.zeros(epochs)

    min_error = 1e+15
    best_iter = 0

    best_w2 = np.zeros([layers_size[1], layers_size[2]])
    best_w1 = np.zeros([layers_size[0], layers_size[1]])

    best_b2 = np.zeros(layers_size[2])
    best_b1 = np.zeros(layers_size[1])

    alpha.set_value(learning_rate)
    train_examples = trainX.shape[0]
    no_batches = train_examples // batch_size

    for epoch in range(epochs):
        if epoch % 100 == 0:
            print('Starting iteration', epoch)

        trainX, trainY = shuffle_data(trainX, trainY)
        cost = 0
        for batch in range(no_batches):
            start, end = batch * batch_size, (batch + 1) * batch_size
            if batch == no_batches - 1:
                end = train_examples
            cost += train(trainX[start:end], trainY[start:end])*1.0*(end-start+1)
        train_cost[epoch] = cost*1.0 / train_examples

        pred, test_cost[epoch], test_accuracy[epoch] = test(testX, testY)
        if test_cost[epoch] < min_error:
            best_iter = epoch
            min_error = test_cost[epoch]
            best_w1 = w1.get_value()
            best_b1 = b1.get_value()
            best_w2 = w2.get_value()
            best_b2 = b2.get_value()

    # set weights and biases to values at which performance was best
    w2.set_value(best_w2)
    b2.set_value(best_b2)
    w1.set_value(best_w1)
    b1.set_value(best_b1)

    best_pred, best_cost, best_accuracy = test(testX, testY)
    print('Minimum error: %.1f, Best accuracy: %.1f, Number of Iterations: %d' % (best_cost, best_accuracy, best_iter))

    return {
        'best_cost': best_cost,
        'train_cost': train_cost,
        'test_cost': test_cost
    }

