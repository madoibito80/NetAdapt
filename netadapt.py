#coding: utf-8

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import optimizers
from chainer import cuda
from chainer.dataset import convert
from chainer import training
from chainer.training import extensions
from chainer.training import triggers
import time
import pickle
import shutil
import sys
import cupy as cp

gpu = 0
batch_size = 128
objective_compression_rate = 0.9
numstep = 20
numlayer = 12


def ChooseNumFilters(model, k, lookup_table, pace):

    (conv_pw1, bn_pw, conv_dw, bn_dw, conv_pw2) = model.predictor.getLayer(k)
    base_numfilter = conv_dw.W.data.shape[1]

    if base_numfilter == 1:
        return False

    base_time = lookup_table[k-2][base_numfilter-1]

    for i in range(1,base_numfilter):
        n = base_numfilter-i
        diff = base_time - lookup_table[k-2][n-1]

        if diff > pace:
            return n

    return 1



def ChooseWhichFilters(conv_pw1, bn_pw, conv_dw, bn_dw, conv_pw2, numfilter):

    # new conv_dw
    param = conv_dw.W.data
    shape = param.shape
    param = param.reshape(shape[1],shape[0]*shape[2]*shape[3])
    param = cuda.to_cpu(param)

    norms = np.linalg.norm(param, axis=1)
    order = np.argsort(norms)[::-1]
    param = np.delete(param, order[numfilter:], axis=0)

    param = param.reshape((shape[0], numfilter, shape[2], shape[3]))
    param = cuda.to_gpu(param)


    new_conv_dw = L.DepthwiseConvolution2D(
        in_channels=numfilter,
        channel_multiplier=1,
        ksize=3,
        stride=conv_dw.stride,
        pad=1,
        nobias=True)

    new_conv_dw.W.data = param


    # new bn_dw
    new_bn_dw = L.BatchNormalization(numfilter)

    param = bn_dw.beta.data
    param = cuda.to_cpu(param)
    param = np.delete(param, order[numfilter:])
    param = cuda.to_gpu(param)
    new_bn_dw.beta.data = param


    param = bn_dw.gamma.data
    param = cuda.to_cpu(param)
    param = np.delete(param, order[numfilter:])
    param = cuda.to_gpu(param)
    new_bn_dw.gamma.data = param

    new_bn_dw.to_gpu(0)


    # new conv_pw1
    param = conv_pw1.W.data
    param = cuda.to_cpu(param)
    param = np.delete(param, order[numfilter:], axis=0)
    param = cuda.to_gpu(param)


    new_conv_pw1 = L.Convolution2D(
        in_channels=conv_pw1.W.data.shape[1],
        out_channels=numfilter,
        ksize=1,
        stride=conv_pw1.stride,
        pad=0,
        nobias=True)

    new_conv_pw1.W.data = param



    # new bn_pw
    new_bn_pw = L.BatchNormalization(numfilter)

    param = bn_pw.beta.data
    param = cuda.to_cpu(param)
    param = np.delete(param, order[numfilter:])
    param = cuda.to_gpu(param, 0)
    new_bn_pw.beta.data = param

    param = bn_pw.gamma.data
    param = cuda.to_cpu(param)
    param = np.delete(param, order[numfilter:])
    param = cuda.to_gpu(param, 0)
    new_bn_pw.gamma.data = param

    new_bn_pw.to_gpu(0)


    # new conv_pw2
    param = conv_pw2.W.data
    param = cuda.to_cpu(param)
    param = np.delete(param, order[numfilter:], axis=1)
    param = cuda.to_gpu(param)


    new_conv_pw2 = L.Convolution2D(
        in_channels=numfilter,
        out_channels=conv_pw2.W.data.shape[0],
        ksize=1,
        stride=conv_pw1.stride,
        pad=0,
        nobias=True)

    new_conv_pw2.W.data = param


    return (new_conv_pw1, new_bn_pw, new_conv_dw, new_bn_dw, new_conv_pw2)




class dw_sep_conv(chainer.Chain):
    def __init__(self, in_c, out_c, stride):
        super(dw_sep_conv, self).__init__()
        with self.init_scope():

            self.conv_dw = L.DepthwiseConvolution2D(in_c, 1, 3, stride=stride, pad=1, nobias=True)
            self.bn_dw = L.BatchNormalization(in_c)

            self.conv_pw = L.Convolution2D(in_c, out_c, 1, stride=1, pad=0, nobias=True)
            self.bn_pw = L.BatchNormalization(out_c)

    def __call__(self, x):
        h = self.conv_dw(x)
        h = F.relu(self.bn_dw(h))
        h = self.conv_pw(h)
        h = F.relu(self.bn_pw(h))
        return h



class MobileNet(chainer.Chain):
    def __init__(self):
        super(MobileNet, self).__init__()
        with self.init_scope():

            self.conv0=L.Convolution2D(3, 32, ksize=3, stride=1, pad=1, nobias=True)
            self.bn0=L.BatchNormalization(32)

            self.block1 = dw_sep_conv(32, 64, 1)
            self.block2 = dw_sep_conv(64, 128, 1)
            self.block3 = dw_sep_conv(128, 128, 1)
            self.block4 = dw_sep_conv(128, 256, 2)
            self.block5 = dw_sep_conv(256, 256, 1)
            self.block6 = dw_sep_conv(256, 512, 2)

            self.block7 = dw_sep_conv(512, 512, 1)
            self.block8 = dw_sep_conv(512, 512, 1)
            self.block9 = dw_sep_conv(512, 512, 1)
            self.block10 = dw_sep_conv(512, 512, 1)
            self.block11 = dw_sep_conv(512, 512, 1)

            self.block12 = dw_sep_conv(512, 1024, 2)
            self.block13 = dw_sep_conv(1024, 1024, 2)

            self.fc13 = L.Linear(1024,10)

            
    def __call__(self, x):

        h = self.bn0(self.conv0(x))

        for i in range(1,14):
            h = self['block{}'.format(i)](h)

        h = F.average_pooling_2d(h, ksize=2, stride=1)
        h = self.fc13(h)

        return h


    def getLayer(self, k):
        return (self['block{}'.format(k-1)].conv_pw,
                self['block{}'.format(k-1)].bn_pw,
                self['block{}'.format(k)].conv_dw,
                self['block{}'.format(k)].bn_dw,
                self['block{}'.format(k)].conv_pw)


    def setLayer(self, k, conv_pw1, bn_pw, conv_dw, bn_dw, conv_pw2):
        self['block{}'.format(k-1)].conv_pw = conv_pw1
        self['block{}'.format(k-1)].bn_pw = bn_pw
        self['block{}'.format(k)].conv_dw = conv_dw
        self['block{}'.format(k)].bn_dw = bn_dw
        self['block{}'.format(k)].conv_pw = conv_pw2




def train_run(model=L.Classifier(MobileNet()), epoch=50):

    (train, test) = chainer.datasets.get_cifar10()

    if gpu >= 0:
        model.to_gpu(gpu)

    optimizer = chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(5e-4))

    train_iter = chainer.iterators.SerialIterator(train, batch_size)
    test_iter = chainer.iterators.SerialIterator(test, batch_size,
                                                 repeat=False, shuffle=False)


    stop_trigger = (epoch, 'epoch')

    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=gpu)

    trainer = training.Trainer(updater, stop_trigger, out='./')

    trainer.extend(extensions.Evaluator(test_iter, model, device=gpu))

    # Reduce the learning rate by half every 25 epochs.
    trainer.extend(extensions.ExponentialShift('lr', 0.5),
                   trigger=(25, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    trainer.extend(extensions.ProgressBar())

    trainer.run()

    return model



def save_model(model):
    f = open('./model.pickle', 'wb')
    pickle.dump(model, f)
    f.close()


def measure_time(conv_pw1, bn_pw, conv_dw, bn_dw, conv_pw2, rep_size, on_cpu=True):

    if on_cpu:
        x = np.zeros((8,conv_pw1.W.data.shape[1],rep_size,rep_size)).astype(np.float32)
        conv_pw1.to_cpu()
        bn_pw.to_cpu()
        conv_dw.to_cpu()
        bn_dw.to_cpu()
        conv_pw2.to_cpu()

    else:
        x = cp.zeros((batch_size,conv_pw1.W.data.shape[1],rep_size,rep_size)).astype(cp.float32)
    
    x = chainer.Variable(x)

    start = time.perf_counter()
    
    for i in range(10):
        h = conv_pw1(x)
        h = F.relu(bn_pw(h))
        h = conv_dw(h)
        h = F.relu(bn_dw(h))
        h = conv_pw2(h)

    elapsed_time = time.perf_counter()-start
    elapsed_time /= 10.0
    return elapsed_time



def make_table(model, on_cpu=True):

    numlayer = 12
    lookup_table = []
    rep_sizes = [32, 32, 32, 32, 16, 16, 8, 8, 8, 8, 8, 8, 4]


    # gpu warmup run
    if on_cpu:
        x = np.zeros((8,3,32,32)).astype(np.float32)
        model.to_cpu()
    else:
        x = cp.zeros((batch_size,3,32,32)).astype(cp.float32)
    x = chainer.Variable(x)

    for i in range(20):
        start = time.perf_counter()
        y = model.predictor(x)
        elapsed_time = time.perf_counter()-start
        print(elapsed_time)
    
    base_time = elapsed_time


    for k in range(2,numlayer+2):
        print(k)

        maxfilter = model.predictor['block{}'.format(k)].conv_pw.W.data.shape[1]

        table_row = np.zeros(maxfilter)

        for n in range(maxfilter):
          
            (conv_pw1, bn_pw, conv_dw, bn_dw, conv_pw2) = model.predictor.getLayer(k)
            (conv_pw1, bn_pw, conv_dw, bn_dw, conv_pw2) = ChooseWhichFilters(conv_pw1, bn_pw, conv_dw, bn_dw, conv_pw2, maxfilter-n)

            table_row[maxfilter-n-1] = measure_time(conv_pw1, bn_pw, conv_dw, bn_dw, conv_pw2, rep_size=rep_sizes[k-1])

        lookup_table.append(table_row)
        print(table_row)

    f = open('./lookup_table.pickle', 'wb')
    pickle.dump(lookup_table, f)
    pickle.dump(base_time, f)
    f.close()


    return lookup_table



def search():


    (train, test) = chainer.datasets.get_cifar10()
    test_iter = chainer.iterators.SerialIterator(test, 32)


    # load lookup table
    f = open('./lookup_table.pickle', 'rb')
    lookup_table = pickle.load(f)
    base_time = pickle.load(f)
    f.close()

    objective_time = base_time * objective_compression_rate
    pace = (base_time - objective_time) / numstep



    shutil.copyfile("./mobilenet.pickle","./copybase.pickle")


    for s in range(numstep):

        best_acc = -float('inf')

        for k in range(2,numlayer+2):

            f = open('./copybase.pickle', 'rb')
            candidate = pickle.load(f)
            f.close()

            n = ChooseNumFilters(candidate, k, lookup_table, pace)
            if not n:
                continue
            print('layer = '+str(k)+', numfilter = '+str(n))
            (conv_pw1, bn_pw, conv_dw, bn_dw, conv_pw2) = candidate.predictor.getLayer(k)
            (conv_pw1, bn_pw, conv_dw, bn_dw, conv_pw2) = ChooseWhichFilters(conv_pw1, bn_pw, conv_dw, bn_dw, conv_pw2, n)
            candidate.predictor.setLayer(k, conv_pw1, bn_pw, conv_dw, bn_dw, conv_pw2)

            train_run(candidate, epoch=0.2)

            acc = 0.0
            for i in range(4):
                batch = test_iter.next()
                x_array, t_array = convert.concat_examples(batch, gpu)
                x = chainer.Variable(x_array)
                t = chainer.Variable(t_array)
                y = candidate.predictor(x)
                acc += cuda.to_cpu(F.accuracy(y,t).data)

            acc /= 4

            print('accuracy = '+str(acc))

            candidate.cleargrads()

            if acc > best_acc:
                best_acc = acc
                f = open('./best_candidate.pickle', 'wb')
                pickle.dump(candidate, f)
                f.close()

        shutil.copyfile("./best_candidate.pickle", "./copybase.pickle")


    f = open('./copybase.pickle', 'rb')
    model = pickle.load(f)
    f.close()



    total_reduc = 0.0

    for k in range(2,numlayer+2):
        n = model.predictor['block{}'.format(k)].conv_dw.W.data.shape[1]
        print('layer'+str(k)+', numfilter='+str(n))
        total_reduc += (lookup_table[k-2][-1] - lookup_table[k-2][n-1])

    print('reduction = '+str(total_reduc/base_time))
    train_run(model, 4)


    f = open('./adapted_model.pickle', 'wb')
    pickle.dump(model, f)
    f.close()


if __name__ == '__main__':

    if sys.argv[1] == 'pretrain':
        model = train_run()
        save_model(model)

    if sys.argv[1] == 'maketable':
        f = open('./mobilenet.pickle', 'rb')
        model = pickle.load(f)
        f.close()
        make_table(model)

    if sys.argv[1] == 'search':
        search()