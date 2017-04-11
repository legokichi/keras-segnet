from typing import Tuple, List, Text, Dict, Any, Iterator, Union
import time
import sys
sys.path.append("/usr/local/Cellar/opencv3/3.2.0/lib/python3.5/site-packages/") # mac opencv path
import cv2
import numpy as np
np.random.seed(1337) # for reproducibility
import os
#os.environ['KERAS_BACKEND'] = 'theano'
#os.environ["THEANO_FLAGS"] = "exception_verbosity=high,optimizer=None,device=cpu"
#os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=cpu,floatX=float32,optimizer=fast_compile'
os.environ['KERAS_BACKEND'] = 'tensorflow'
import random


import keras
# keras.backend.backend()
# keras.backend.set_epsilon(1e-07)
# keras.backend.epsilon()
# keras.backend.set_floatx('float32')
# keras.backend.floatx()
# keras.backend.set_image_data_format('channels_first') # theano
# keras.backend.image_data_format()
from keras.preprocessing.image import ImageDataGenerator

import SegNet

# for CamVid
# https://github.com/alexgkendall/SegNet-Tutorial/blob/master/Models/segnet_train.prototxt
CLASS_WEIGHT = [0.2595, 0.1826, 4.5640, 0.1417, 0.5051, 0.3826, 9.6446, 1.8418, 6.6823, 6.2478, 3.0, 7.3614]

# for CamVid
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

    label_colours = np.array([ Sky, Building, Pole, Road, Pavement, Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled ])
    return label_colours

# for CamVid
def create_batch(batch_size: int=8, nb_class: int=12, ignored: int=11) -> Iterator[Tuple[np.ndarray, np.ndarray]] :
    '''
    channels_last
    '''
    with open('./SegNet-Tutorial/CamVid/train.txt', 'r') as f:
        lines = f.readlines()
    pairs = [tuple(line.strip().replace('/SegNet', './SegNet-Tutorial').split(' ', 1)) for line in lines] # type: List[Tuple[str, str]]
    while True:
        selected = random.sample(pairs, batch_size) # type: List[Tuple[str, str]]
        loaded = [(preprocess_input(x), preprocess_teacher(y, nb_class, ignored)) for (x, y) in selected] # type: List[Tuple[np.ndarray, np.ndarray]]
        _x = np.array([x for (x, y) in loaded]) # (n, 480, 360, 3)
        _y = np.array([y for (x, y) in loaded]) # (n, 480, 360, nb_class)
        yield (_x, _y)


def preprocess_input(filename: str) -> np.ndarray:
    img = np.einsum('hwc->whc', cv2.imread(filename))
    _img = normalized(img)
    return _img

def preprocess_teacher(filename: str, nb_class: int, ignored: int) -> np.ndarray:
    '''
    labeling
    '''
    # label は RGB すべてに 0~255 の範囲のクラス値が入っている
    img = np.einsum('hw->wh', cv2.imread(filename, cv2.IMREAD_GRAYSCALE).astype(np.int32) )
    (w, h) = img.shape # == (480, 360)
    # https://github.com/pfnet-research/chainer-segnet/blob/ca84cd694351eeaff357656e76baa310dc455e66/lib/camvid.py#L63
    img[np.where(img == ignored)] = -1 # ラベル ignored は Unlabelled
    _img = np.zeros((w, h, nb_class), dtype=np.int8) # == (480, 360, nb_class)
    # https://github.com/pradyu1993/segnet/blob/master/segnet.py#L50
    for i in range(w):
        for j in range(h):
            _img[i, j, img[i][j]] = 1
    return _img


def normalized(rgb: np.ndarray) -> np.ndarray:
    '''
    equalizeHist for RGB
    return rgb/255.0
    '''
    norm=np.zeros((rgb.shape[0], rgb.shape[1], 3),np.float32)

    b=rgb[:,:,0]
    g=rgb[:,:,1]
    r=rgb[:,:,2]

    norm[:,:,0]=cv2.equalizeHist(b)
    norm[:,:,1]=cv2.equalizeHist(g)
    norm[:,:,2]=cv2.equalizeHist(r)

    return norm


def train():
    model = SegNet.train(
        shape=(480, 360, 3), 
        nb_class=12,
        batch_gen=create_batch(8),
        class_weight=CLASS_WEIGHT
    )
    predict(model)



if __name__ == '__main__':
    train()
    exit()


def predict(model: Union[keras.engine.training.Model, None]):
    if model == None: 
        start = time.time()
        model = SegNet.load()
        end = time.time()
        print('%30s' % 'load_weights in ', str((end - start)*1000), 'ms')


    start = time.time()
    frame = np.einsum('hwc->whc', normalized(cv2.imread("SegNet-Tutorial/CamVid/test/Seq05VD_f02370.png")))
    end = time.time()
    print('%30s' % 'imread in ', str((end - start)*1000), 'ms')

    start = time.time()
    output = model.predict_proba(frame)
    end = time.time()
    print('%30s' % 'predict_proba in ', str((end - start)*1000), 'ms')

    start = time.time()
    labeled = np.argmax(output[0], axis=1)
    img = np.einsum('whc->hwc', visualize(labeled))
    cv2.imwrite("output.png", img)
    end = time.time()
    print('%30s' % 'imwrite in ', str((end - start)*1000), 'ms')



def visualize(labeled: np.ndarray) -> np.ndarray:
    '''
    labeled: (w, h, c=0~11)
    '''
    r = labeled.copy()
    g = labeled.copy()
    b = labeled.copy()
    label_colours = create_label_colors()

    for l in range(0,11):
        r[labeled==l]=label_colours[l,0]
        g[labeled==l]=label_colours[l,1]
        b[labeled==l]=label_colours[l,2]

    rgb = np.zeros((labeled.shape[0], labeled.shape[1], 3))
    rgb[:,:,0] = (r/255.0)#[:,:,0]
    rgb[:,:,1] = (g/255.0)#[:,:,1]
    rgb[:,:,2] = (b/255.0)#[:,:,2]
    return rgb # (w, h, c)

