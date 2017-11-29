import sys
import pickle
import numpy as np
from numpy.random import *
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils.np_utils import to_categorical

factors = np.load('../auto/201307-201406input.npy')
train_factors , test_factors = factors[:200], factors[200:]
N = np.load('../auto/output.npy')
train_N, test_N = N[:200], N[200:]

model = Sequential()
model.add(Dense(output_dim=100, input_dim=234))
model.add(Dense(output_dim=655))
model.add(Activation("sigmoid"))
model.add(BatchNormalization())
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.fit(train_factors,train_N, nb_epoch=1000, batch_size=16)  # Train
print('finish')
model.fit(train_factors,train_N, nb_epoch=1000, batch_size=16)  # Train
print('finish')

PRED = model.predict_proba(test_factors, batch_size=10)

for t,p in zip(test_N,PRED):
    print('One day ...')
    print(t) ; print(t.shape)
    print(p) ; print(p.shape)
