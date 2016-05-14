#!/usr/bin/python

import numpy as np
import time
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import data

class MlpLearning(ChainList):
    def __init__(self, units):
        super(MlpLearning, self).__init__()
        if len(units) < 3:
            raise ValueError('units must be equal or over 3 layers')
        for (a, b) in zip(units[:-1], units[1:]):
            self.add_link(L.Linear(a, b))
        self._dropout = False

    def layer_size(self):
        return len(self)

    def units(self):
        u = []
        for i in range(self.layer_size()):
            u.append(len(self[i].W.data[0]))
        u.append(len(self[i].W.data))
        return u

    def norm(self):
        return map(lambda x: np.linalg.norm(x.W.data), self)

    def forward(self, input):
        h = input
        for i in range(self.layer_size()-1):
            h = F.relu(self[i](h))
#            h = F.dropout(F.relu(self[i](h)), train= not input.volatile)
        return self[-1](h)

    def __call__(self, input):
        return self.forward(input)

class MnistData(object):
    def __init__(self):
        self._data = data.load_mnist_data()
        self._data['data'] = self._data['data'].astype(np.float32)
        self._data['data'] /= 255
        self._data['target'] = self._data['target'].astype(np.int32)
        self._last = 0

    def take(self, size):
        d = self._data['data'][self._last:self._last+size]
        t = self._data['target'][self._last:self._last+size]
        self._last += size
        return (d, t)

    def x_dimension(self):
        return len(self._data['data'][0])

    def y_range(self):
        return max(self._data['target'])+1

class Trainer(object):
    def __init__(self, data):
        self._data = data
        self.n_epoch = 4
        self.n_training = 200
        self.n_test = 25
        self.n_batch = 15
        self.mlp = None
        self.train_loss = []
        self.train_accuracy = []
        self.test_loss = []
        self.test_accuracy = []        

    def setup(self, epoch=4, training=200, test=25, batch=15):
        self.n_epoch = epoch
        self.n_training = training
        self.n_test = test
        self.n_batch = batch

    def create_mlp(self, hidden_nunits):
        l0 = self._data.x_dimension()
        lz = self._data.y_range()
        self.mlp = MlpLearning([l0] + hidden_nunits + [lz])
        return self.mlp


    def learn(self, mlp):
        
        x_train, y_train = self._data.take(self.n_training)
        x_test, y_test = self._data.take(self.n_test)
        model = chainer.links.Classifier(mlp)
        optimizer = chainer.optimizers.Adam()
        optimizer.setup(model)
        self.train_loss = []
        self.train_accuracy = []
        self.test_loss = []
        self.test_accuracy = []
        for e in range(self.n_epoch):
            self.training(optimizer, model, x_train, y_train)
            self.test(model, x_test, y_test)

    def training(self, optimizer, model, x_train, y_train):
        sum_loss = 0
        sum_accuracy = 0
        start = time.time()
        perm = np.random.permutation(self.n_training)
        for i in range(0, self.n_training, self.n_batch):
            x = chainer.Variable(np.asarray(x_train[perm[i:i+self.n_batch]]))
            t = chainer.Variable(np.asarray(y_train[perm[i:i+self.n_batch]]))
            optimizer.update(model, x, t)
            sum_loss += float(model.loss.data) * len(t.data)
            sum_accuracy += float(model.accuracy.data) * len(t.data)
        self.train_loss.append(sum_loss / self.n_training)
        self.train_accuracy.append(sum_accuracy / self.n_training)
        print('train mean loss={}, accuracy={}, elapsed time={} sec'.format(
            sum_loss / self.n_training, sum_accuracy / self.n_training,
            time.time() - start))

    def test(self, model, x_test, y_test):
        sum_loss = 0
        sum_accuracy = 0
        for i in range(0, self.n_test, self.n_batch):
            x = chainer.Variable(np.asarray(x_test[i:i+self.n_batch]),
                                 volatile='on')
            t = chainer.Variable(np.asarray(y_test[i:i+self.n_batch]),
                                 volatile='on')
            model(x, t)
            sum_loss += float(model.loss.data) * len(t.data)
            sum_accuracy += float(model.accuracy.data) * len(t.data)
        self.test_loss.append(sum_loss / self.n_test)
        self.test_accuracy.append(sum_accuracy / self.n_test)

        print('test mean loss={}, accuracy={}'.format(
            sum_loss / self.n_test, sum_accuracy / self.n_test))
