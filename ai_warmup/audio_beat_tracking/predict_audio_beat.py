from __future__ import print_function
import librosa
import numpy as np
import os
import sys
import argparse

import keras
from keras.models import Sequential, load_model
from keras.layers import LSTM, TimeDistributed, Dense, Activation, Dropout
from keras.optimizers import RMSprop, Adam, SGD
from keras import backend as K

audio_raw_data, sr = librosa.load(train_fprefix +'.wav', sr=sr, mono=True)

model = load_model("./dataset/log/models/model_lr_0.000001.hdf5")
pred = model.predict(X_batch, BATCH_SIZE)
predict(self, x, batch_size=32, verbose=0)