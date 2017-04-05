import os
#os.environ['KERAS_BACKEND'] = 'theano'
#os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=cpu,floatX=float32,optimizer=fast_compile'

from keras.backend import set_image_dim_ordering
set_image_dim_ordering('tf')

import sys
sys.path.append("/usr/local/Cellar/opencv3/3.2.0/lib/python3.5/site-packages/")
import cv2

import time
import numpy as np
np.random.seed(1337) # for reproducibility

import keras.models as models
from keras.layers.noise import GaussianNoise
from keras.layers.core import Activation, Reshape, Permute, Lambda
from keras.layers.convolutional import Conv2D, UpSampling2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD


Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road_marking = [255,69,0]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

label_colours = np.array([
  Sky, Building, Pole, Road, Pavement,
  Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist,
  Unlabelled])

def visualize(temp):
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0,11):
        r[temp==l]=label_colours[l,0]
        g[temp==l]=label_colours[l,1]
        b[temp==l]=label_colours[l,2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:,:,0] = (r/255.0)#[:,:,0]
    rgb[:,:,1] = (g/255.0)#[:,:,1]
    rgb[:,:,2] = (b/255.0)#[:,:,2]
    return rgb

def normalized(rgb):
    '''
    equalizeHist
    '''
    #return rgb/255.0
    norm=np.zeros((rgb.shape[0], rgb.shape[1], 3),np.float32)

    b=rgb[:,:,0]
    g=rgb[:,:,1]
    r=rgb[:,:,2]

    norm[:,:,0]=cv2.equalizeHist(b)
    norm[:,:,1]=cv2.equalizeHist(g)
    norm[:,:,2]=cv2.equalizeHist(r)

    return norm

def binarylab(labels):
    x = np.zeros([360,480,12])
    for i in range(360):
        for j in range(480):
            x[i,j,labels[i][j]]=1
    return x

def prep_data():
    train_data = []
    train_label = []
    txt = ""
    with open(os.path.join(PATH, 'train.txt')) as f:
        txt = f.readlines()
        txt = [line.split(' ') for line in txt]
    print(txt)
    for i in range(len(txt)):
        data, label = txt[i]
        data, label = (data[15:], label[15:]) # /SegNet/CamVid/train/0001TP_006690.png -> train/0001TP_006690.png
        label = label[:-1] # remove \n
        data, label = ( os.path.join(PATH, data), os.path.join(PATH, label) )
        train_data.append(np.rollaxis(normalized(cv2.imread(data)),2))
        train_label.append(binarylab(cv2.imread(label)[:,:,0]))
        print('.',end='')
    return np.array(train_data), np.array(train_label)

def create_encoding_layers():
    return [
        ZeroPadding2D(padding=(1, 1)),
        Conv2D(64, (3, 3), padding='valid'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),

        ZeroPadding2D(padding=(1, 1)),
        Conv2D(128, (3, 3), padding='valid'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),

        ZeroPadding2D(padding=(1, 1)),
        Conv2D(256, (3, 3), padding='valid'),
        BatchNormalization(),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),

        ZeroPadding2D(padding=(1, 1)),
        Conv2D(512, (3, 3), padding='valid'),
        BatchNormalization(),
        Activation('relu'),
        #MaxPooling2D(pool_size=(2, 2),
    ]

def create_decoding_layers():
    pool_size = 2
    return[
        #UpSampling2D(size=(pool_size,pool_size)),
        ZeroPadding2D(padding=(1, 1)),
        Conv2D(512, (3, 3), padding='valid'),
        BatchNormalization(),

        UpSampling2D(size=(pool_size,pool_size)),
        ZeroPadding2D(padding=(1, 1)),
        Conv2D(256, (3, 3), padding='valid'),
        BatchNormalization(),

        UpSampling2D(size=(pool_size,pool_size)),
        ZeroPadding2D(padding=(1, 1)),
        Conv2D(128, (3, 3), padding='valid'),
        BatchNormalization(),

        UpSampling2D(size=(pool_size,pool_size)),
        ZeroPadding2D(padding=(1, 1)),
        Conv2D(64, (3, 3), padding='valid'),
        BatchNormalization(),
    ]

def create_segnet(channels, height, width):
    segnet = models.Sequential()

    # Add a noise layer to get a denoising segnet. This helps avoid overfitting
    segnet.add(Lambda(lambda x: x + 0, input_shape=(height, width, channels) ))

    #segnet.add(GaussianNoise(sigma=0.3))
    
    for l in create_encoding_layers(): segnet.add(l)
    for l in create_decoding_layers(): segnet.add(l)


    segnet.add(Conv2D(12, (1, 1), padding='valid',))
    segnet.add(Reshape((12, height*width)))
    segnet.add(Permute((2, 1)))
    segnet.add(Activation('softmax'))

    return segnet


PATH = './SegNet-Tutorial/CamVid/'

CLASS_WEIGHTING = [0.2595, 0.1826, 4.5640, 0.1417, 0.5051, 0.3826, 9.6446, 1.8418, 6.6823, 6.2478, 3.0, 7.3614]
NB_EPOCH = 100
BATCH_SIZE = 14

if __name__ == '__main__':
    train_data, train_label = prep_data()
    #train_label = np.reshape(train_label, (367, 360*480, 12))

    segnet = create_segnet(3, 360, 480)
    segnet.summary()

    #optimizer = SGD(lr=0.01, momentum=0.8, decay=0., nesterov=False)
    segnet.compile(loss="categorical_crossentropy", optimizer='adadelta')

    '''
    history = segnet.fit(
      train_data,
      train_label,
      batch_size=BATCH_SIZE,
      nb_epoch=NB_EPOCH,
      show_accuracy=True,
      verbose=1,
      class_weight=CLASS_WEIGHTING
      # validation_data=(X_test, X_test),
    )
    segnet.save_weights('model_weight_ep100.hdf5')
    '''
    
    start = time.time()
    segnet.load_weights('segnet/model_weight_ep100.hdf5')
    end = time.time()
    print('%30s' % 'load_weights in ', str((end - start)*1000), 'ms')


    start = time.time()
    frame = np.rollaxis(normalized(cv2.imread("SegNet-Tutorial/CamVid/test/Seq05VD_f02370.png")))
    end = time.time()
    print('%30s' % 'imread in ', str((end - start)*1000), 'ms')

    start = time.time()
    output = segnet.predict_proba(frame)
    end = time.time()
    print('%30s' % 'predict_proba in ', str((end - start)*1000), 'ms')

    start = time.time()
    pred = visualize(np.argmax(output[0],axis=1).reshape((360,480)), False)
    cv2.imwrite("output.png", pred)
    end = time.time()
    print('%30s' % 'imwrite in ', str((end - start)*1000), 'ms')
