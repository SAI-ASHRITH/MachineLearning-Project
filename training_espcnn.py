import sys
import keras
import cv2
import numpy
import matplotlib
import skimage
from tensorflow.keras import layers
import tensorflow as tf

# import the necessary packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.optimizers import Adam
from skimage.measure import compare_ssim as ssim
from keras import backend as K
from keras.callbacks import LearningRateScheduler
from matplotlib import pyplot as plt
import cv2
import numpy as np
import math
import os
import random


def train_ESPCNN():
    espcnn=get_model()
    random.seed()
    degraded=[]
    ref=[]

    count=0
    for file in os.listdir(os.getcwd()+'/train-bsd/'):
        
        ref_e = cv2.imread(os.getcwd()+'/train-bsd/{}'.format(file))
        ref_e = cv2.cvtColor(ref_e, cv2.COLOR_BGR2YUV)
        #print(np.shape(ref_e))
        ref_e=ref_e[:,:,0]
        ref_e=ref_e[10:310,10:310]

        h = ref_e.shape[0]
        w = ref_e.shape[1]
        
        new_height = h // 3
        new_width = w // 3
        #print(w)
        #print(new_width)
        deg_e=cv2.resize(ref_e,(new_width,new_height),cv2.INTER_LINEAR)
 
        print(np.shape(ref_e))
        print(np.shape(deg_e))
        ref_e=ref_e.astype(float)/255
        deg_e=deg_e.astype(float)/255
        temp2=np.zeros((np.shape(ref_e)[0],np.shape(ref_e)[1],1))
        temp1=np.zeros((np.shape(deg_e)[0],np.shape(deg_e)[1],1))
        temp1[:,:,0]=deg_e
        temp2[:,:,0]=ref_e
        
        degraded.append(temp1)
        ref.append(temp2)

        count+=1
   
    count=0       
    print(np.shape(degraded))
    print(np.shape(ref))
    randd=list(zip(degraded,ref)) 
    random.shuffle(randd)
    degraded,ref=zip(*randd)    

    early_stopping_callback = keras.callbacks.EarlyStopping(monitor="loss", patience=10)
    loss_fn = keras.losses.MeanSquaredError()
    optimizer = Adam(learning_rate=0.001)
    callbacks = [ early_stopping_callback]
    espcnn.compile(optimizer=optimizer, loss="mean_squared_error")

    espcnn.fit(np.array(degraded),np.array(ref),epochs=500,shuffle=True,  verbose=2)
    espcnn.save("trained_espcnn_new.h5")    
    
def get_model(upscale_factor=3, channels=1):
    conv_args = {
        "activation": "relu",
        "kernel_initializer": "Orthogonal",
        "padding": "same",
    }
    inputs = keras.Input(shape=(None, None, channels))
    x = layers.Conv2D(64, 5, **conv_args)(inputs)
    x = layers.Conv2D(64, 3, **conv_args)(x)
    x = layers.Conv2D(32, 3, **conv_args)(x)
    x = layers.Conv2D(channels * (upscale_factor ** 2), 3, **conv_args)(x)
    outputs = tf.nn.depth_to_space(x, upscale_factor)

    return keras.Model(inputs, outputs)    

train_ESPCNN()    