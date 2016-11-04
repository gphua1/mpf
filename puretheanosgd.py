from __future__ import print_function
import matplotlib.pyplot as pp

import os
import sys
import timeit
import math

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
print(np.random.randint(16))


def symmetra(n):
    W = np.zeros((n, n))

    for i in range(n):
        for j in range(n):

            if i < j:
                W[i, j] = np.random.normal(0, 1)
                W[j, i] = W[i, j]
    return W


class dA(object):

    def __init__(self, input=None, W=None, b=None):

        if not W:

            initial_W = np.asarray(symmetra(16), dtype=theano.config.floatX)
            print(initial_W)
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not b:
            initial_b = np.asarray(np.zeros((1, 16)), dtype=theano.config.floatX)
            b = theano.shared(value=initial_b, name='b', borrow=True)

        self.W = W
        self.b = b

        if input is None:

            initial_x = T.dmatrix(name='input')
            x = theano.shared(value=initial_x, name='x', borrow=True)
            self.x = x

        else:
            self.x = input

        self.params = [self.W, self.b]

    def get_cost_updates(self, learning_rate):

        nsamp = 10
        eps = 0.01
        delt = 0.5 - self.x
        cost =   (eps/nsamp)*T.sum(T.exp((delt * (T.dot(self.x, self.W) + self.b.reshape([1, -1])))))
        print(cost.shape)
        gparams = T.grad(cost, self.params)

        gparams[0]=theano.tensor.extra_ops.fill_diagonal(gparams[0], 0)
        gparams[0]=gparams[0]+gparams[0].T

        updates = [
            (param, param - learning_rate * gparam)

            for param, gparam in zip(self.params, gparams)
            ]

        return (cost, updates, gparams)


def test_dA(learning_rate=0.01, training_epochs=1000,
            batch_size=10):


    datasets = np.load('DataSync51.dat.npy')
    datasets=datasets.T
    datasets=datasets[1000:51000,:]

    print(datasets.shape)
    train_set_x = datasets
    train_set_x2 = theano.shared(value=train_set_x, name='train_set_x2', borrow=True)

    n_train_batches = train_set_x.shape[0] / batch_size

    index = T.lscalar()
    x = T.matrix('x')

    da = dA(input=x)

    ndata = datasets.shape[0]

    cost, updates, gparams = da.get_cost_updates(learning_rate=learning_rate)

    train_da = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x2[index * batch_size: (index + 1) * batch_size]
        }
    )

    grad_da = theano.function(
        [index],
        gparams,
        updates=updates,
        givens={
            x: train_set_x2[index * batch_size: (index + 1) * batch_size]
        }
    )

    for epoch in range(training_epochs):
        c = []

        for batch_index in range(n_train_batches):
            c.append(train_da(batch_index))
            g = (grad_da(batch_index))



        print('Training epoch %d, cost ' % epoch, np.mean(c))

    return da.W.get_value(borrow=True), g


if __name__ == '__main__':
    Wmpf, g = test_dA()


    x, y = Wmpf.shape
    Wreal = np.load('WSync51.dat.npy')

    M = math.sqrt(np.sum(np.square(Wmpf)))
    R = math.sqrt(np.sum(np.square(Wreal)))
    Wmpf = Wmpf
    Wreal = Wreal

    mse = np.sum(np.square(Wreal - Wmpf)) / (x * y)
    print('mse')
    print(mse)
    WmpfLine = np.ravel(Wmpf)
    WrealLine = np.ravel(Wreal)

    print(WmpfLine[0:10])
    print(WrealLine[0:10])
    np.save('Wtheano.dat', Wmpf)
    np.save('Wgrad.dat', g)

    pp.plot(WmpfLine[0:50])
    pp.plot(WrealLine[0:50])
    pp.show()