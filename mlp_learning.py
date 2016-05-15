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

    def output_parameters(self):
        print('#layer:{}'.format(self.mlp.units()))
        #print('#total W:{}'.format(self.mlp.n_parameter_w()))
        #print('#total b:{}'.format(self.mlp.n_parameter_b()))
        print('#epoch:{}'.format(self.n_epoch))
        print('#training data:{}'.format(self.n_training))
        #print('#validation data:{}'.format("0"))
        print('#test data:{}'.format(self.n_test))
        print('#batch size:{}'.format(self.n_batch))
        #print('#dropout:{}'.format("No"))

    def learn(self, mlp):
        self.output_parameters()
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
        loss_accu = self.batch_loop()
        perm = np.random.permutation(self.n_training)
        for i in range(0, self.n_training, self.n_batch):
            x = chainer.Variable(np.asarray(x_train[perm[i:i+self.n_batch]]))
            t = chainer.Variable(np.asarray(y_train[perm[i:i+self.n_batch]]))
            optimizer.update(model, x, t)
            loss_accu.update(model.loss.data, model.accuracy.data, len(t.data))
        self.train_loss.append(loss_accu.loss_mean())
        self.train_accuracy.append(loss_accu.accuracy_mean())
        loss_accu.output('train mean')

    def test(self, model, x_test, y_test):
        loss_accu = self.batch_loop()
        for i in range(0, self.n_test, self.n_batch):
            x = chainer.Variable(np.asarray(x_test[i:i+self.n_batch]),
                                 volatile='on')
            t = chainer.Variable(np.asarray(y_test[i:i+self.n_batch]),
                                 volatile='on')
            model(x, t)
            loss_accu.update(model.loss.data, model.accuracy.data, len(t.data))
        self.test_loss.append(loss_accu.loss_mean())
        self.test_accuracy.append(loss_accu.accuracy_mean())
        loss_accu.output('test mean')

    def batch_loop(self):
        return BatchLoop()

class TrainerQuiet(Trainer):
    def __init__(self, data):
        super(TrainerQuiet, self).__init__(data)

    def batch_loop(self):
        return BatchLoopQuiet()

    def output_parameters(self):
        pass

class BatchLoop(object):
    def __init__(self):
        self._loss = 0
        self._accuracy = 0
        self._size = 0
        self._start = time.time()

    def update(self, loss, accuracy, size):
        self._loss += float(loss) * size
        self._accuracy += float(accuracy) * size
        self._size += size

    def loss_mean(self):
        return self._loss / self._size

    def accuracy_mean(self):
        return self._accuracy / self._size

    def elapse_time(self):
        return self._start / time.time()

    def output(self, header):
        print('{} loss={}, accuracy={}, elapsed_time={}'.format(
            header, self._loss/self._size, self._accuracy/self._size,
            self.elapse_time()))

class BatchLoopQuiet(BatchLoop):
    def __init__(self):
        super(BatchLoopQuiet, self).__init__()

    def output(self, header):
        pass
