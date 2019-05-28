from numpy.random import seed
seed(101)
from tensorflow import set_random_seed
set_random_seed(101)

from pprint import pprint
import pandas as pd
import numpy as np
import keras
from keras.utils.np_utils import to_categorical # used for converting labels to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras import backend as K


import tensorflow as tf
#from tf.keras import layers

import os

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import itertools
import shutil
import matplotlib.pyplot as plt
#%matplotlib inline

pprint(os.listdir('F:\\Documents\\cancer-project\\input'))

base_dir = 'base_dir'
#os.mkdir(base_dir)

train_dir = os.path.join(base_dir, 'train_dir')
#os.mkdir(train_dir)

val_dir = os.path.join(base_dir, 'val_dir')
#os.mkdir(val_dir)

# create new folders inside train_dir
nv = os.path.join(train_dir, 'nv')
#os.mkdir(nv)
mel = os.path.join(train_dir, 'mel')
#os.mkdir(mel)
bkl = os.path.join(train_dir, 'bkl')
#os.mkdir(bkl)
bcc = os.path.join(train_dir, 'bcc')
#os.mkdir(bcc)
akiec = os.path.join(train_dir, 'akiec')
#os.mkdir(akiec)
vasc = os.path.join(train_dir, 'vasc')
#os.mkdir(vasc)
df = os.path.join(train_dir, 'df')
#os.mkdir(df)

# create new folders inside val_dir
nv = os.path.join(val_dir, 'nv')
#os.mkdir(nv)
mel = os.path.join(val_dir, 'mel')
#os.mkdir(mel)
bkl = os.path.join(val_dir, 'bkl')
#os.mkdir(bkl)
bcc = os.path.join(val_dir, 'bcc')
#os.mkdir(bcc)
akiec = os.path.join(val_dir, 'akiec')
#os.mkdir(akiec)
vasc = os.path.join(val_dir, 'vasc')
#os.mkdir(vasc)
df = os.path.join(val_dir, 'df')
#os.mkdir(df)


df_data = pd.read_csv('F:\\Documents\\cancer-project\\input\\HAM10000_metadata.csv')

print(df_data.head())