import sys
import keras
import cv2
import numpy
import matplotlib
import skimage
print('Python: {}'.format(sys.version))
print('Keras: {}'.format(keras.__version__))
print('OpenCV: {}'.format(cv2.__version__))
print('NumPy: {}'.format(numpy.__version__))
print('Matplotlib: {}'.format(matplotlib.__version__))
print('Scikit-Image: {}'.format(skimage.__version__))
# import the necessary packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.optimizers import Adam
from skimage.measure import compare_ssim as ssim
from matplotlib import pyplot as plt
import cv2
import numpy as np
import math
import os
import random

# python magic function, displays pyplot figures in the notebook
#%matplotlib inline
# define a function for peak signal-to-noise ratio (PSNR)
def psnr(target, ref):
         
    # assume RGB image
    target_data = target.astype(float)
    ref_data = ref.astype(float)

    diff = ref_data - target_data
    diff = diff.flatten('C')

    rmse = math.sqrt(np.mean(diff ** 2.))

    return 20 * math.log10(255. / rmse)

# define function for mean squared error (MSE)
def mse(target, ref):
    # the MSE between the two images is the sum of the squared difference between the two images
    err = np.sum((target.astype('float') - ref.astype('float')) ** 2)
    err /= float(target.shape[0] * target.shape[1])
    
    return err

# define function that combines all three image quality metrics
def compare_images(target, ref):
    scores = []
    scores.append(psnr(target, ref))
    scores.append(mse(target, ref))
    scores.append(ssim(target, ref, multichannel =True))
    
    return scores

# prepare degraded images by introducing quality distortions via resizing



# define the SRCNN model
def model():
    
    # define model type
    SRCNN = Sequential()
    
    # add model layers
    SRCNN.add(Conv2D(filters=128, kernel_size = (9, 9), kernel_initializer='glorot_uniform',
                     activation='relu', padding='valid', use_bias=True, input_shape=(33, 33, 1)))
    SRCNN.add(Conv2D(filters=64, kernel_size = (3, 3), kernel_initializer='glorot_uniform',
                     activation='relu', padding='same', use_bias=True))
    #SRCNN.add(Conv2D(filters=32, kernel_size = (3, 3), kernel_initializer='glorot_uniform',
    #                 activation='relu', padding='same', use_bias=True))                 
    SRCNN.add(Conv2D(filters=1, kernel_size = (5, 5), kernel_initializer='glorot_uniform',
                     activation='linear', padding='valid', use_bias=True))
    
    # define optimizer
    adam = Adam(lr=0.0003)
    
    # compile model
    SRCNN.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])
    
    return SRCNN


# define necessary image processing functions

def modcrop(img, scale):
    tmpsz = img.shape
    sz = tmpsz[0:2]
    sz = sz - np.mod(sz, scale)
    img = img[0:sz[0], 0:sz[1]]
    return img


def shave(image, border):
    img = image[border: -border, border: -border]
    return img

# define main prediction function



def train():
    srcnn=model()
    degraded_vali=[]
    ref_vali=[]
    degraded=[]
    ref=[]
    count=0
    for file in os.listdir(os.getcwd()+'/train91'):
        
        ref_e = cv2.imread(os.getcwd()+'/train91/{}'.format(file))
        ref_e = cv2.cvtColor(ref_e, cv2.COLOR_BGR2YCrCb)
        ref_e=ref_e[:,:,0]
        #ref_e=cv2.normalize(ref_e[:,:,0], None, 0.0, 1.0, cv2.NORM_MINMAX)
        ref_e=modcrop(ref_e,3)
        h = ref_e.shape[0]
        w = ref_e.shape[1]
        new_height = h // 2
        new_width = w // 2
 
        deg_e=cv2.resize(cv2.resize(ref_e,(new_width,new_height)),(w,h))
        ref_e=ref_e.astype(float)/255
        deg_e=deg_e.astype(float)/255
        temp1=np.zeros((33,33,1))
        temp2=np.zeros((21,21,1))
 
        for x in range(0,ref_e.shape[0]-33,14):
            for y in range(0,ref_e.shape[1]-33,14):
                temp1[:,:,0]=deg_e[x:x+33,y:y+33]
                temp2[:,:,0]=ref_e[x+6:x+27,y+6:y+27]
                
                degraded.append(temp1)
                ref.append(temp2)
                count+=1
    print(np.shape(degraded)) 
    print(np.shape(ref)) 
    count=0       

    randd=list(zip(degraded,ref))
    random.shuffle(randd)
    degraded,ref=zip(*randd)        

    srcnn.fit(np.array(degraded),np.array(ref) ,epochs=200,shuffle=True, batch_size=128, verbose=1)
    srcnn.save("trainedsrcnn.h5")

train()

