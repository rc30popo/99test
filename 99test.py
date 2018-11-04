# -*- Coding: utf-8 -*-

# Numpy
import numpy as np
from numpy.random import *
# Chainer
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Chain,optimizers,Variable

# Parameters
n_epoch = 200

# Neural Network

class DNN(Chain):
    def __init__(self):
        super(DNN, self).__init__(
            l1 = L.Linear(None,100),
            l2 = L.Linear(None,100),
            l3 = L.Linear(None,100)
        )
    def forward(self,x0,x1):
        h = F.relu(self.l1(F.concat((x0,x1),axis=1)))
        h = F.relu(self.l2(h))
        h = self.l3(h)
        return h

x0_train = []
x1_train = []
t_train = []

for x0 in range(10):
    x0_vec = [0] * 10
    x0_vec[x0] = 1
    for x1 in range(10):
        x1_vec = [0] * 10
        x1_vec[x1] = 1
        x0_train.append(x0_vec)
        x1_train.append(x1_vec)
        t_train.append(x0 * x1)

x0_train = np.array(x0_train,dtype=np.float32)
x1_train = np.array(x1_train,dtype=np.float32)
t_train = np.array(t_train,dtype=np.int32)

# Create DNN class instance
model = DNN()

# Set optimizer
optimizer = optimizers.Adam()
optimizer.setup(model)

# Training
for epoch in range(n_epoch):
    perm = np.random.permutation(len(x0_train))
    x0v = Variable(x0_train[perm])
    x1v = Variable(x1_train[perm])
    t = Variable(t_train[perm])
    y = model.forward(x0v,x1v)
    model.cleargrads()
    loss = F.softmax_cross_entropy(y, t)
    loss.backward()
    optimizer.update()
    print("epoch: {}, mean loss: {}".format(epoch, loss.data))

# Execute test
ok_cnt = 0
for x0 in range(10):
    for x1 in range(10):
        x0_vec = [0] * 10
        x0_vec[x0] = 1
        x1_vec = [0] * 10
        x1_vec[x1] = 1
        x0_test = []
        x1_test = []
        x0_test.append(x0_vec)
        x1_test.append(x1_vec)
        x0_test = np.array(x0_test,dtype=np.float32)
        x1_test = np.array(x1_test,dtype=np.float32)
        x0v = Variable(x0_test)
        x1v = Variable(x1_test)
        y = model.forward(x0v,x1v)
        y = np.argmax(y.data[0])
        match = False
        if y == x0 * x1:
            ok_cnt += 1
            match = True
        print("{} * {} = Predicted {}, Expected {},Match {}".format(x0,x1,y,x0 * x1,match))

print("Ok {}/Total {}".format(ok_cnt,100))
