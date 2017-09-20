import numpy as np
import os
import sys
import argparse

a = [0., 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1]
b = [0 , 0 ,   0,   1,   1,   1,   1,   1,   1,   1,   1,  1,   1]
c = np.equal(b, np.round(np.clip(a, 0, 1)))
print(c)

'''
parser = argparse.ArgumentParser()
parser.add_argument('-epoch', type=int, default=200, help='the number of times to iterate over the training data')
parser.add_argument('-lr', type=lambda lr_str: [int(lr) for lr in lr_str.split(',')], default=[0.001], help='learning rate')
args = parser.parse_args()
print(args.epoch)
learning_rates = args.lr
print(learning_rates)

a = np.ndarray([], dtype=int)
a.resize((5,10), refcheck=False)
a.fill(1)
#print(a)

a = np.array([[1,2,3],[4,5,6]])
#a = np.ndarray)
print(a.shape)
print(a)

a.resize(2, 1, 3)
print(a)
print(a[0])
print(a[1])

x = np.array([[1,0,0], [0,2,0], [1,1,0]])
print(x)
xx = np.nonzero(x)
print(xx)
xxx = np.transpose(xx)
print(xxx)

x = [0, 1, 0, 0, 1]
xx = np.nonzero(x)[0]
print(xx)

x[1:3] = [3,2]
print(x)

'''
