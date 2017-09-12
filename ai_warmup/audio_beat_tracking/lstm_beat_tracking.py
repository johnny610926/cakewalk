from __future__ import print_function
import librosa
import numpy as np
import os
import sys
import argparse

import keras
from keras.models import Sequential, load_model
from keras.layers import LSTM, TimeDistributed, Dense, Activation, Dropout
from keras.optimizers import RMSprop, Adam
from keras import backend as K
from keras.callbacks import ModelCheckpoint

from sklearn.preprocessing import MinMaxScaler

from mirex2016_dataset import load_trainXY

def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    _precision = true_positives / (predicted_positives + K.epsilon())
    _recall = true_positives / (possible_positives + K.epsilon())
    _f1_score = 2 * (_precision * _recall) / (_precision + _recall)
    return _f1_score

def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    _precision = true_positives / (predicted_positives + K.epsilon())
    return _precision

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    _recall = true_positives / (possible_positives + K.epsilon())
    return _recall

def __create_dataset(train_x, train_y, look_back): # deplicated
    n_of_frames, n_of_freq_bins = train_x.shape
    _train_x, _train_y = [], []
    for i in range(n_of_frames - look_back):
        _train_x.append(train_x[i:i+look_back, :])
        _train_y.append(train_y[i+look_back-1, 0])
    return np.array(_train_x), np.array(_train_y)

def create_dataset(trainX, trainY, n_of_frames, look_back, diff_halfwave=False):
    total_frames, n_of_freq_bins = trainX.shape

    if diff_halfwave:
        trainX = np.diff(trainX, axis=0)
        trainX = np.concatenate((trainX[:1, :], trainX), axis=0)
        trainX = np.maximum(np.zeros(trainX.shape), trainX)
        scaler = MinMaxScaler(feature_range=(0, 1))
        trainX = scaler.fit_transform(trainX)

    _trainX, _trainY = [], []
    for s in range(total_frames // n_of_frames):
        frame_start = s * n_of_frames
        for i in range(n_of_frames):
            if i < look_back:
                train_x = np.zeros((look_back, n_of_freq_bins))
                train_x[-(i+1):, :] = trainX[frame_start:(frame_start + i + 1), :]
                _trainX.append(train_x)
            else:
                _trainX.append(trainX[(frame_start + i - look_back):(frame_start + i), :])
            _trainY.append(trainY[frame_start + i - 1, 0])
    return np.array(_trainX), np.array(_trainY)

parser = argparse.ArgumentParser()
#x431x128_y40x431
parser.add_argument('--datadir', type=str, default='./dataset/mirex_beat_tracking_2016/train/x862x128_y862x1/', help='train data path')
parser.add_argument('--logdir', type=str, default='./dataset/log', help='Tensorboard log path')
parser.add_argument('--wtdir', type=str, default='./dataset/log/models', help='Weights checkpoint log path')
parser.add_argument('--epochs', type=int, default=500, help='the number of times to iterate over the training data')
parser.add_argument('-lr', type=lambda lr_str: [int(lr) for lr in lr_str.split(',')],
    default=[0.001, 0.0001, 0.00001, 0.000001], help='learning rate')
parser.add_argument('--look_back', type=int, default=20, help='look back the number of frame to predict the beat. This value will be assigned to "time_steps"')
args = parser.parse_args()
trainXY_path = args.datadir
tb_logdir = args.logdir
wt_logdir = args.wtdir
epochs = args.epochs
learning_rates = args.lr
look_back_frames = args.look_back

trainX, trainY, n_of_frames, n_of_freq_bins = load_trainXY(trainXY_path)
#print(str(trainX.shape) +', '+ str(trainY.shape) +', '+ str(n_of_frames) +', '+ str(n_of_freq_bins))

trainX, trainY = create_dataset(trainX, trainY, n_of_frames, look_back_frames, diff_halfwave=True)
#print(str(trainX.shape) +', '+ str(trainY.shape))

# Total 20 songs, 16 for traings, 4 for validation
x_val, y_val = trainX[n_of_frames * 16:, :, :], trainY[n_of_frames * 16:]
x_train, y_train = trainX[:n_of_frames * 16, :, :], trainY[:n_of_frames * 16]

loss_function = 'binary_crossentropy'
for lr in learning_rates:
    model = Sequential()
    # build a LSTM RNN
    model.add(LSTM(
        units=128,
        batch_input_shape=(n_of_frames, look_back_frames, n_of_freq_bins),
        activation='tanh', #'relu',
        recurrent_activation='hard_sigmoid', #'relu',
        use_bias=True,
        return_sequences=True,     # False: Many(X) to One(Y). True: Many(X) to Many(Y)
        stateful=True,              # True: the final state of batch1 is feed into the initial state of batch2
    ))
    model.add(Dropout(0.3))
    model.add(LSTM(40, return_sequences=False, stateful=True))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='hard_sigmoid'))
    #optimizer = RMSprop(lr=lr)
    optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy', recall, precision])

    tb_logpath = tb_logdir + '/tanh_adam_lr_%f' % (lr)
    tb = keras.callbacks.TensorBoard(log_dir=tb_logpath, histogram_freq=0, write_graph=True, write_images=True)
    wt_logpath = wt_logdir + '/model_lr_%f.hdf5' % (lr)
    ch_pt = ModelCheckpoint(filepath=wt_logpath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False)
    model.fit(x_train, y_train,
        batch_size=n_of_frames,
        epochs=epochs,
        validation_data=(x_val, y_val),
        shuffle=False,
        verbose=0,
        callbacks=[tb, ch_pt])
    #model.reset_states()
