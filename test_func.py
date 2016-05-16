
import unittest
from subprocess import Popen, PIPE

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

    def test_null(self):
        output = self.cmdline("")
        self.assertTrue(output.find("epoch:") >= 0)

    def test_epoch(self):
        output = self.cmdline("--epoch 0")
        self.assertTrue(output.find("epoch:0") >= 0)
        self.assertEqual(output.find("test mean"), -1)

    def test_training_size(self):
        output = self.cmdline("--epoch 0 --trainingsize 500")
        self.assertTrue(output.find("training data size:500") >= 0)
        
    def test_test_size(self):
        output = self.cmdline("--epoch 0 --testsize 100")
        self.assertTrue(output.find("test data size:100") >= 0)
        
if __name__ == '__main__':
    unittest.main()
