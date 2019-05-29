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

#print(os.listdir("C:\Users\davyd\Documents\input"))

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
#s.mkdir(bkl)
bcc = os.path.join(val_dir, 'bcc')
#os.mkdir(bcc)
akiec = os.path.join(val_dir, 'akiec')
#os.mkdir(akiec)
vasc = os.path.join(val_dir, 'vasc')
#os.mkdir(vasc)
df = os.path.join(val_dir, 'df')
#os.mkdir(df)

#Crear data de train y val
df_data = pd.read_csv('F:\\Descargas\\cancer-project\\input\\HAM10000_metadata.csv')

#pprint(df_data.head())

df = df_data.groupby('lesion_id').count()

df = df[df['image_id'] == 1]

df.reset_index(inplace=True)

#print(df)

#Identificar imagenes duplicadas

def identify_duplicates(x):
    
    unique_list = list(df['lesion_id'])
    
    if x in unique_list:
        return 'no_duplicates'
    else:
        return 'has_duplicates'

df_data['duplicates'] = df_data['lesion_id']

df_data['duplicates'] = df_data['duplicates'].apply(identify_duplicates)

#print(df_data)

#Excluir imagenes repetidas

df = df_data[df_data['duplicates'] == 'no_duplicates']

#print(df)

#Creo def_val para el conjunto de datos Val

y = df['dx']

_, df_val = train_test_split(df, test_size=0.17, random_state=101, stratify=y)

#print(df_val)

#print(df_val['dx'].value_counts())

#Crear data para Train

def identify_val_rows(x):
    # create a list of all the lesion_id's in the val set
    val_list = list(df_val['image_id'])
    
    if str(x) in val_list:
        return 'val'
    else:
        return 'train'

# identify train and val rows

# create a new colum that is a copy of the image_id column
df_data['train_or_val'] = df_data['image_id']
# apply the function to this new column
df_data['train_or_val'] = df_data['train_or_val'].apply(identify_val_rows)
   
# filter out train rows
df_train = df_data[df_data['train_or_val'] == 'train']


#print(df_train)
#print(df_val)

df_data.set_index('image_id', inplace=True)

folder_1 = os.listdir('F:\\Descargas\\cancer-project\\input\\ham10000_images_part_1')
folder_2 = os.listdir('F:\\Descargas\\cancer-project\\input\\ham10000_images_part_2')

train_list = list(df_train['image_id'])
val_list = list(df_val['image_id'])

print(len(train_list))
print(len(val_list))

print(len(os.listdir('base_dir/train_dir/nv')))
print(len(os.listdir('base_dir/train_dir/mel')))
print(len(os.listdir('base_dir/train_dir/bkl')))
print(len(os.listdir('base_dir/train_dir/bcc')))
print(len(os.listdir('base_dir/train_dir/akiec')))
print(len(os.listdir('base_dir/train_dir/vasc')))
print(len(os.listdir('base_dir/train_dir/df')))