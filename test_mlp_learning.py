import unittest
import numpy as np
import numpy.random as npr
import chainer
import data

from mlp_learning import MlpNet, MnistData, Trainer, BatchLoop
from mlp_learning import TrainerQuiet, BatchLoopQuiet

RandomSeed = 53269 #for regression test

class TestMlpNet(unittest.TestCase):
    def test_init(self):
        with self.assertRaises(TypeError):
            a = MlpNet(0,1,2)

    def test_init2(self):
        mlp = MlpNet([5,3,2])

    def test_init3(self):
        with self.assertRaises(ValueError):
            mlp = MlpNet([3,2])

    def test_init4(self):
        with self.assertRaises(ValueError):
            mlp = MlpNet([10,-1,10])

    def test_init5(self):
        with self.assertRaises(ZeroDivisionError):
            mlp = MlpNet([10,0,10])

    def test_layer_len1(self):
        mlp = MlpNet([5,3,4])
        self.assertEqual(len(mlp), 2)

    def test_layer_len2(self):
        mlp = MlpNet([10,5,5,10])
        self.assertEqual(len(mlp), 3)

    def test_norm(self):
        npr.seed(RandomSeed)
        mlp = MlpNet([5, 5, 5])
        self.assertEqual(len(mlp.norm()), 2)
        self.assertAlmostEqual(mlp.norm()[0], 2.1161823)
        self.assertAlmostEqual(mlp.norm()[1], 1.6219993)

    def test_forward(self):
        npr.seed(RandomSeed)
        x_data = np.array(npr.randn(1, 3), dtype=np.float32)
        x = chainer.Variable(x_data)
        mlp = MlpNet([3,2,4])
        mlp.zerograds()
        y = mlp.forward(x)
        self.assertAlmostEqual(np.linalg.norm(y.data), 1.6243998)
        y.grad = np.ones((1,4), dtype=np.float32)
        y.backward()
        self.assertAlmostEqual(np.linalg.norm(x.grad), 0.22090262)

    def test_load_data(self):
        mnist = data.load_mnist_data()
        self.assertEqual(len(mnist['data']), 70000)
        self.assertEqual(len(mnist['data'][0]), 784)
        self.assertEqual(mnist['data'][0][0], 0)
        self.assertEqual(mnist['data'][0][156], 126)
        self.assertEqual(len(mnist['target']), 70000)
        self.assertEqual(mnist['target'][0], 5)

    def load_mnist_binary_data(self):
        mnist = data.load_mnist_data()
        mnist['data'] = mnist['data'].astype(np.float32)
        mnist['data'] /= 255
        mnist['target'] = mnist['target'].astype(np.int32)
        return mnist
        
    def test_load_binary_data(self):
        mnist = self.load_mnist_binary_data()
        self.assertEqual(len(mnist['data']), 70000)
        self.assertAlmostEqual(mnist['data'][0][156], 126.0/255)
        self.assertEqual(mnist['target'][0], 5)

    def test_batch(self):
        npr.seed(RandomSeed)
        N = 25
        mnist = self.load_mnist_binary_data()
        x = chainer.Variable(np.asarray(mnist['data'][:N]))
        t = chainer.Variable(np.asarray(mnist['target'][:N]))
        mlp = MlpNet([784,100,100,10])
        model = chainer.links.Classifier(mlp)
        loss = model(x, t)
        sum_loss = float(loss.data)
        sum_accuracy = float(model.accuracy.data)
        self.assertAlmostEqual(sum_loss, 2.3380165100097656)
        self.assertEqual(sum_accuracy, 0)

    def test_training(self):
        npr.seed(RandomSeed)
        N = 25
        mnist = self.load_mnist_binary_data()
        x = chainer.Variable(np.asarray(mnist['data'][:N]))
        t = chainer.Variable(np.asarray(mnist['target'][:N]))
        mlp = MlpNet([784,100,100,10])
        model = chainer.links.Classifier(mlp)
        optimizer = chainer.optimizers.Adam()
        optimizer.setup(model)
        optimizer.update(model, x, t)
        self.assertAlmostEqual(float(model.accuracy.data), 0)
        optimizer.update(model, x, t)
        self.assertAlmostEqual(float(model.accuracy.data), 0.20000000298023224)
        optimizer.update(model, x, t)
        self.assertAlmostEqual(float(model.accuracy.data), 0.6000000238418579)
        optimizer.update(model, x, t)
        self.assertAlmostEqual(float(model.accuracy.data), 0.8799999952316284)
        optimizer.update(model, x, t)
        self.assertAlmostEqual(float(model.loss.data), 1.9289497137069702)
        self.assertAlmostEqual(float(model.accuracy.data), 0.9200000166893005)
#        self.assertEqual(mlp[2].b.data, [0,0])

    def test_epoch(self):
        npr.seed(RandomSeed)
        TrainSize = 75
        BatchSize = 20
        TestSize = 15
        mnist = MnistData()
        x_train, y_train = mnist.take(TrainSize)
        x_test, y_test = mnist.take(TestSize)
        mlp = MlpNet([784,100,100,10])
        model = chainer.links.Classifier(mlp)
        optimizer = chainer.optimizers.Adam()
        optimizer.setup(model)
        for i in range(0, TrainSize, BatchSize):
            x = chainer.Variable(np.asarray(x_train[i:i+BatchSize]))
            t = chainer.Variable(np.asarray(y_train[i:i+BatchSize]))
            optimizer.update(model, x, t)
        self.assertAlmostEqual(float(model.loss.data), 2.2083351612091064)
        self.assertAlmostEqual(float(model.accuracy.data), 0.13333334028720856)
        x = chainer.Variable(np.asarray(x_test))
        t = chainer.Variable(np.asarray(y_test))
        model(x, t)
        self.assertAlmostEqual(float(model.loss.data), 2.1313300132751465)
        self.assertAlmostEqual(float(model.accuracy.data), 0.4000000059604645)

    def test_mnist_data(self):
        mnist = MnistData()
        x_train, y_train = mnist.take(1000)
        self.assertEqual(len(x_train), 1000)
        self.assertEqual(len(x_train[0]), 784)
        self.assertEqual(min(x_train[0]), 0)
        self.assertEqual(max(x_train[0]), 1)
        self.assertEqual(len(y_train), 1000)
        self.assertEqual(min(y_train), 0)
        self.assertEqual(max(y_train), 9)
        x_test, y_test = mnist.take(200)
        self.assertNotEqual(x_train[0][156], x_test[0][156])

    def test_mnist_data_dimension(self):
        mnist = MnistData()
        self.assertEqual(mnist.x_dimension(), 784)

    def test_mnist_data_y_range(self):
        mnist = MnistData()
        self.assertEqual(mnist.y_range(), 10)

    def test_mlp_units(self):
        mlp = MlpNet([784,100,100,10])
        self.assertEqual(mlp.units(), [784,100,100,10])

    def test_trainer(self):
        args = {}
        trainer = TrainerQuiet(MnistData(), args)
        mlp = trainer.create_mlp([112, 112])
        self.assertEqual(mlp.units(), [784, 112, 112, 10])
        trainer.setup(epoch=4, training=200, test=25, batch=15)
        trainer.learn(mlp)
        self.assertAlmostEqual(trainer.train_accuracy[-1], 0.9000000059604645)
        self.assertAlmostEqual(trainer.test_accuracy[-1], 0.880000007153)

    def test_batch_loop(self):
        loop = BatchLoop()

if __name__ == '__main__':
    unittest.main()
