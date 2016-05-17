
import unittest
from subprocess import Popen, PIPE
import os

import numpy as np
import numpy.random as npr
import chainer
import data

from mlp_learning import MlpLearning, MnistData, Trainer, BatchLoop
from mlp_learning import TrainerQuiet, BatchLoopQuiet

class TestMlpLearning(unittest.TestCase):
    def cmdline(self, arg_string):
        commands = ("python mlp_learning.py "+arg_string).split()
        return Popen(commands, stdout=PIPE).communicate()[0]

    def test_help(self):
        output = self.cmdline("--help")
        head = "usage: mlp_learning.py [-h]"
        output = output[:len(head)]
        self.assertEqual(output, head)

    def test_default(self):
        output = self.cmdline("--epoch 0")
        self.assertTrue(output.find("epoch:0") >= 0)
        self.assertTrue(output.find("#layer:[784, 112, 112, 10]") >= 0)
        self.assertTrue(output.find("#number of parameters:101706") >= 0)
        self.assertTrue(output.find("#training data size:60000") >= 0)
        self.assertTrue(output.find("#test data size:10000") >= 0)
        self.assertTrue(output.find("#batch size:100") >= 0)
        
        self.assertEqual(output.find("test mean"), -1)

    def test_training_size(self):
        output = self.cmdline("--epoch 0 --trainingsize 500")
        self.assertTrue(output.find("training data size:500") >= 0)
        
    def test_test_size(self):
        output = self.cmdline("--epoch 0 --testsize 100")
        self.assertTrue(output.find("test data size:100") >= 0)
        
    def test_layer(self):
        output = self.cmdline("--epoch 0 --unit 100 80")
        self.assertTrue(output.find("layer:[784, 100, 80, 10]") >= 0)
        self.assertTrue(output.find("number of parameters:87390") >= 0)

    def test_save_model(self):
        self.cmdline("--epoch 1 --training 1000 --test 100")
        model = open("mlp.model","r")
        model.close()
        state = open("mlp.state", "r")
        state.close()
        os.remove("mlp.model")
        os.remove("mlp.state")

    def train_accuracy(self, log):
        lines = log.split('\n')
        accuracy = []
        for l in lines:
            if l.find('train mean') == 0:
                for item in l.split(' '):
                    if item.find('accuracy=') == 0:
                        accuracy.append(float(item[len('accuracy='):].strip(',')))
        return accuracy[-1]

    def test_initmodel(self):
        result = self.cmdline("--epoch 1 --training 1000 --test 100")
        accuracy = self.train_accuracy(result)
        self.assertLess(accuracy, 0.5)
        result = self.cmdline("--epoch 1 --training 1000 --test 100 --initmodel mlp.model")
        accuracy = self.train_accuracy(result)
        self.assertGreater(accuracy, 0.5)

if __name__ == '__main__':
    unittest.main()
