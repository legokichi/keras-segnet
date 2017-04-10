from typing import Tuple, List, Text, Dict, Any
import time
import sys
sys.path.append("/usr/local/Cellar/opencv3/3.2.0/lib/python3.5/site-packages/")
import cv2
import numpy as np
np.random.seed(1337) # for reproducibility
import os
os.environ['KERAS_BACKEND'] = 'theano'
os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=cpu,floatX=float32,optimizer=fast_compile'
# os.environ['KERAS_BACKEND'] = 'tensorflow'

import keras
# keras.backend.backend()
# keras.backend.set_epsilon(1e-07)
# keras.backend.epsilon()
# keras.backend.set_floatx('float32')
# keras.backend.floatx()
keras.backend.set_image_data_format('channels_first') # theano
# keras.backend.image_data_format()

from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten
from keras.layers.core import Dense, Reshape, Permute
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential, Model
from keras.layers.convolutional import Conv2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

class DePool2D(UpSampling2D):
    '''
    https://github.com/nanopony/keras-convautoencoder/blob/c8172766f968c8afc81382b5e24fd4b57d8ebe71/autoencoder_layers.py#L24
    Simplar to UpSample, yet traverse only maxpooled elements
    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.
    # Output shape
        4D tensor with shape:
        `(samples, channels, upsampled_rows, upsampled_cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, upsampled_rows, upsampled_cols, channels)` if dim_ordering='tf'.
    # Arguments
        size: tuple of 2 integers. The upsampling factors for rows and columns.
        dim_ordering: 'th' or 'tf'.
            In 'th' mode, the channels dimension (the depth)
            is at index 1, in 'tf' mode is it at index 3.
    '''

    def __init__(self, pool2d_layer, *args, **kwargs):
        self._pool2d_layer = pool2d_layer
        super().__init__(*args, **kwargs)

    def get_output(self, train=False):
        X = self.get_input(train)
        if self.dim_ordering == 'th':
            output = keras.backend.repeat_elements(X, self.size[0], axis=2)
            output = keras.backend.repeat_elements(output, self.size[1], axis=3)
        elif self.dim_ordering == 'tf':
            output = keras.backend.repeat_elements(X, self.size[0], axis=1)
            output = keras.backend.repeat_elements(output, self.size[1], axis=2)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        f = keras.backend.gradients(
            keras.backend.sum(
                self._pool2d_layer.get_output(train)
            ),
            self._pool2d_layer.get_input(train)
        ) * output

        return f

def create_segnet(shape=(None, 3, 224, 244)) -> keras.engine.training.Model :
    # input_shape: (include_top is False のときのみ) 
    # ex. (3, 224, 244) or (224, 224, 3)
    # 正確に3つの入力チャンネルを持つ必要があり、幅と高さは48以上でなければなりません。
    input_tensor = Input(shape=shape) # type: object
    encoder = VGG16(
        include_top=False, 
        weights='imagenet', 
        input_tensor=input_tensor,
        input_shape=shape,
        pooling="None" ) # type: keras.engine.training.Model
    encoder.summary()

    L = [layer for i, layer in enumerate(encoder.layers) ] # type: List[keras.engine.topology.Layer]
    for layer in L: layer.trainable = False # freeze VGG16
    L.reverse()

    x = encoder.output

    # Block 5
    x = DePool2D(L[0], size=L[0].pool_size, input_shape=encoder.output_shape[1:])(x)
    x = Activation('relu')(BatchNormalization()(Conv2D(L[1].filters, L[1].kernel_size, padding=L[1].padding, kernel_initializer="he_normal", bias_initializer='zeros')(x)))
    x = Activation('relu')(BatchNormalization()(Conv2D(L[2].filters, L[2].kernel_size, padding=L[2].padding, kernel_initializer="he_normal", bias_initializer='zeros')(x)))
    x = Activation('relu')(BatchNormalization()(Conv2D(L[3].filters, L[3].kernel_size, padding=L[3].padding, kernel_initializer="he_normal", bias_initializer='zeros')(x)))
    # Block 4
    x = DePool2D(L[4], size=L[4].pool_size)(x)
    x = Activation('relu')(BatchNormalization()(Conv2D(L[5].filters, L[5].kernel_size, padding=L[5].padding, kernel_initializer="he_normal", bias_initializer='zeros')(x)))
    x = Activation('relu')(BatchNormalization()(Conv2D(L[6].filters, L[6].kernel_size, padding=L[6].padding, kernel_initializer="he_normal", bias_initializer='zeros')(x)))
    x = Activation('relu')(BatchNormalization()(Conv2D(L[7].filters, L[7].kernel_size, padding=L[7].padding, kernel_initializer="he_normal", bias_initializer='zeros')(x)))
    # Block 3
    x = DePool2D(L[8], size=L[8].pool_size)(x)
    x = ZeroPadding2D(padding=(0, 1))(x)
    x = Activation('relu')(BatchNormalization()(Conv2D(L[10].filters, L[10].kernel_size, padding=L[10].padding, kernel_initializer="he_normal", bias_initializer='zeros')(x)))
    x = Activation('relu')(BatchNormalization()(Conv2D(L[11].filters, L[11].kernel_size, padding=L[11].padding, kernel_initializer="he_normal", bias_initializer='zeros')(x)))
    # Block 2
    x = DePool2D(L[12], size=L[12].pool_size)(x)
    x = Activation('relu')(BatchNormalization()(Conv2D(L[13].filters, L[13].kernel_size, padding=L[13].padding, kernel_initializer="he_normal", bias_initializer='zeros')(x)))
    x = Activation('relu')(BatchNormalization()(Conv2D(L[14].filters, L[14].kernel_size, padding=L[14].padding, kernel_initializer="he_normal", bias_initializer='zeros')(x)))
    # Block 1
    x = DePool2D(L[15], size=L[15].pool_size)(x)
    x = Activation('relu')(BatchNormalization()(Conv2D(L[16].filters, L[16].kernel_size, padding=L[16].padding, kernel_initializer="he_normal", bias_initializer='zeros')(x)))
    x = Activation('relu')(BatchNormalization()(Conv2D(L[17].filters, L[17].kernel_size, padding=L[17].padding, kernel_initializer="he_normal", bias_initializer='zeros')(x)))

    x = Activation('softmax')(Conv2D(12, (1, 1), padding='valid',)(x))

    predictions = x

    segnet = Model(input=encoder.inputs, outputs=predictions) # type: keras.engine.training.Model
    sgd = SGD(lr=0.01, momentum=0.8, decay=1e-6, nesterov=True)
    segnet.compile(loss="categorical_crossentropy", optimizer=sgd)
    segnet.summary()
    return segnet
    
def train(model: keras.engine.training.Model):

    # we create two instances with the same arguments
    data_gen_args = dict(
                        featurewise_center=True,
                        featurewise_std_normalization=True,
                        rotation_range=90.,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=0.2) # type: dict
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1
    #image_datagen.fit(images, augment=True, seed=seed)
    #mask_datagen.fit(masks, augment=True, seed=seed)
    img_rows = 480
    img_cols = 360
    image_generator = image_datagen.flow_from_directory(
        'SegNet-Tutorial/CamVid/train',
        classes=[],
        class_mode=None,
        target_size=(img_rows, img_cols),
        batch_size=8,
        shuffle=False,
        seed=seed)

    mask_generator = mask_datagen.flow_from_directory(
        'SegNet-Tutorial/CamVid/trainannot',
        class_mode=None,
        classes=[],
        target_size=(img_rows, img_cols),
        batch_size=8,
        shuffle=False,
        #color_mode='grayscale',
        seed=seed)

    # combine generators into one which yields image and masks
    train_generator = zip(image_generator, mask_generator)

    CLASS_WEIGHTING = [0.2595, 0.1826, 4.5640, 0.1417, 0.5051, 0.3826, 9.6446, 1.8418, 6.6823, 6.2478, 3.0, 7.3614]

    segnet.save_weights('model_weight.hdf5')
    print("start")

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=50,
        verbose=1,
        class_weight=CLASS_WEIGHTING
    )




if __name__ == '__main__':
    segnet = create_segnet((3, 480, 360))
    train(segnet)
    '''
    start = time.time()
    segnet.load_weights('segnet/model_weight_ep100.hdf5')
    end = time.time()
    print('%30s' % 'load_weights in ', str((end - start)*1000), 'ms')
    '''


exit()

def predict(model: keras.engine.training.Model):
    start = time.time()
    frame = np.rollaxis(normalized(cv2.imread("SegNet-Tutorial/CamVid/test/Seq05VD_f02370.png")))
    end = time.time()
    print('%30s' % 'imread in ', str((end - start)*1000), 'ms')

    start = time.time()
    output = model.predict_proba(frame)
    end = time.time()
    print('%30s' % 'predict_proba in ', str((end - start)*1000), 'ms')

    start = time.time()
    pred = visualize(np.argmax(output[0],axis=1).reshape((360,480)))
    cv2.imwrite("output.png", pred)
    end = time.time()
    print('%30s' % 'imwrite in ', str((end - start)*1000), 'ms')


def create_label_colors() -> np.ndarray:
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
        Unlabelled
    ])
    return label_colours

def visualize(temp: np.ndarray) -> np.ndarray:
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    label_colours = create_label_colors()

    for l in range(0,11):
        r[temp==l]=label_colours[l,0]
        g[temp==l]=label_colours[l,1]
        b[temp==l]=label_colours[l,2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:,:,0] = (r/255.0)#[:,:,0]
    rgb[:,:,1] = (g/255.0)#[:,:,1]
    rgb[:,:,2] = (b/255.0)#[:,:,2]
    return rgb

def normalized(rgb: np.ndarray) -> np.ndarray:
    #return rgb/255.0
    norm=np.zeros((rgb.shape[0], rgb.shape[1], 3),np.float32)

    b=rgb[:,:,0]
    g=rgb[:,:,1]
    r=rgb[:,:,2]

    norm[:,:,0]=cv2.equalizeHist(b)
    norm[:,:,1]=cv2.equalizeHist(g)
    norm[:,:,2]=cv2.equalizeHist(r)

    return norm

def binarylab(labels: np.ndarray) -> np.ndarray:
    x = np.zeros([360,480,12])
    for i in range(360):
        for j in range(480):
            x[i,j,labels[i][j]]=1
    return x

def prep_data()-> Tuple[np.ndarray, np.ndarray] :
    '''
    train_data, train_label = prep_data()
    train_label = np.reshape(train_label, (367, 360*480, 12))
    '''
    PATH = './SegNet-Tutorial/CamVid/'
    train_data = []
    train_label = []
    txt = [] # type: List[Tuple[Text, Text]]
    with open(os.path.join(PATH, 'train.txt')) as f:
        txt = [line.split(' ') for line in f.readlines()]
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






