#!/usr/bin/python

import numpy as np
import time
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
import data
import argparse

class MlpNet(ChainList):
    def __init__(self, units):
        super(MlpNet, self).__init__()
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

    def nparam(self):
        unitlist = self.units()
        sum = 0
        for i in range(len(unitlist)-1):
            sum += (unitlist[i]+1) * unitlist[i+1]
        return sum

    def norm(self):
        return map(lambda x: np.linalg.norm(x.W.data), self)

    def __call__(self, input): # forward
        h = input
        for i in range(self.layer_size()-1):
            h = F.relu(self[i](h))
#            h = F.dropout(F.relu(self[i](h)), train= not input.volatile)
        return self[-1](h)

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
    def __init__(self, data, args):
        self._data = data
        self._args = args
        self.setup()
        self.mlp = None
        self.init_loss_accu()

    def init_loss_accu(self):
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
        self.mlp = MlpNet([l0] + hidden_nunits + [lz])
        return self.mlp

    def output_parameters(self):
        print('#layer:{}'.format(self.mlp.units()))
        print('#number of parameters:{}'.format(self.mlp.nparam()))
        #print('#total W:{}'.format(self.mlp.n_parameter_w()))
        #print('#total b:{}'.format(self.mlp.n_parameter_b()))
        print('#epoch:{}'.format(self.n_epoch))
        print('#training data size:{}'.format(self.n_training))
        #print('#validation data:{}'.format("0"))
        print('#test data size:{}'.format(self.n_test))
        print('#batch size:{}'.format(self.n_batch))
        #print('#dropout:{}'.format("No"))

    def get_mlp(self, units=None):
        if units:
            mlp = self.create_mlp(units)
        else:
            mlp = self.create_mlp([112,112])
        return mlp

    def get_model(self, mlp):
        model = chainer.links.Classifier(mlp)
        self.load_model(model)
        return model

    def get_optimizer(self, model):
        optimizer = chainer.optimizers.Adam()
        optimizer.setup(model)
        self.load_state(optimizer)
        return optimizer

    def learn(self):
        args = self._args
        if args:
            self.setup(
                epoch = args.epoch, batch = args.batchsize, 
                training = args.trainingsize, test = args.testsize)
            mlp = self.get_mlp(args.unit)
        else:
            mlp = self.get_mlp()

        x_train, y_train = self._data.take(self.n_training)
        x_test, y_test = self._data.take(self.n_test)
        model = self.get_model(mlp)
        optimizer = self.get_optimizer(model)
        self.output_parameters()
        self.init_loss_accu()
        start = time.time()
        for e in range(self.n_epoch):
            self.training(optimizer, model, x_train, y_train)
            accu = self.test(model, x_test, y_test)
        self.output_time(time.time() - start)
        self.save_model_state(model, optimizer)

    def load_model(self, model):
        if 'initmodel' in self._args and self._args.initmodel:
            print '****{}****'.format(self._args.initmodel)
            serializers.load_npz(self._args.initmodel, model)

    def load_state(self, optimizer):
        if 'resume' in self._args and self._args.resume:
            serializers.load_npz(self._args.resume, optimizer)

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
        return loss_accu.accuracy_mean()

    def save_model_state(self, model, optimizer):
        serializers.save_npz('mlp.model', model)
        serializers.save_npz('mlp.state', optimizer)

    def output_time(self, second):
        print('#total elapsed time {:.1f} minitues'.format(second/60))

    def output_save_model(self):
        print('save the model and state')

    def batch_loop(self):
        return BatchLoop()

class TrainerQuiet(Trainer):
    def __init__(self, data, args):
        super(TrainerQuiet, self).__init__(data, args)

    def batch_loop(self):
        return BatchLoopQuiet()

    def output_parameters(self):
        pass

    def output_save_model(self):
        pass

    def output_time(self, second):
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
        return time.time() - self._start

    def output(self, header):
        print('{} loss={}, accuracy={}, elapsed_time={:.1f}'.format(
            header, self._loss/self._size, self._accuracy/self._size,
            self.elapse_time()))

class BatchLoopQuiet(BatchLoop):
    def __init__(self):
        super(BatchLoopQuiet, self).__init__()

    def output(self, header):
        pass

def get_args():
    parser = argparse.ArgumentParser(
        description='Solving MNIST by Multi Layer Perseptron')
    parser.add_argument('--initmodel', '-m', default='',
                        help='Initialize the model from given file')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the optimization from snapshot')
    parser.add_argument('--epoch', '-e', default=4, type=int,
                        help='number of epochs to learn')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='learning minibatch size')
    parser.add_argument('--unit', '-u', type=int, nargs='+',
                        help='number of units')
    parser.add_argument('--trainingsize', '-t', type=int, default=60000,
                        help='training data size')
    parser.add_argument('--testsize', type=int, default=10000,
                        help='test data size')
    return parser.parse_args()

if __name__ == '__main__':
    main_args = get_args()
    dataset = MnistData()
    trainer = Trainer(dataset, main_args)
    trainer.learn()
