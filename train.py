from typing import Tuple, List, Text, Dict, Any, Iterator
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

import keras
# keras.backend.backend()
# keras.backend.set_epsilon(1e-07)
# keras.backend.epsilon()
# keras.backend.set_floatx('float32')
# keras.backend.floatx()
# keras.backend.set_image_data_format('channels_first') # theano
# keras.backend.image_data_format()
from keras.preprocessing.image import ImageDataGenerator

import segnet

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

def read_entry(nb_class = 12) -> Iterator[Tuple[np.ndarray, np.ndarray]] :
    # channels_last
    with open('./SegNet-Tutorial/CamVid/train.txt', 'r') as f:
        while True:
            line = f.readline()
            if not line: break
            _line = line.strip().replace('/SegNet', './SegNet-Tutorial') # type: List[str]
            (x, y) = tuple([np.einsum('hwc->whc', cv2.imread(x)) for x in _line.split(' ', 1)]) # type: Tuple[numpy.ndarray, numpy.ndarray]
            #print(x.shape, y.shape)
            # normalize
            _x = normalized(x)
            (w, h, ch) = y.shape # == (480, 360, 3)
            # y は RGB すべてに 0~255 の範囲のクラス値が入っている
            _y = np.zeros((w, h, nb_class), dtype=np.int8) # == (480, 360, nb_class)
            for i in range(w):
                for j in range(h):
                    _y[i, j, y[i][j][0]] = 1
            #print(x.shape, _y.shape)
            yield (_x, _y)

def create_batch(batch_size: int)-> Iterator[Tuple[np.ndarray, np.ndarray]] :
    while True:
        gen = read_entry()
        i = 0
        while True:
            batch_data = [XY for (_, XY) in zip(range(batch_size), gen)]
            if len(batch_data) == 0: break
            _x = np.array([x for (x, y) in batch_data]) # (n, 480, 360, 3)
            _y = np.array([y for (x, y) in batch_data]) # (n, 480, 360, nb_class)
            if _x.shape[0] != batch_size: break
            print(i, _x.shape, _y.shape)
            i = i + 1
            yield (_x, _y)
        print("recur")




def train(model: keras.engine.training.Model):
    print("start")
    # for (i, (x, y)) in enumerate(create_batch(8)): print(i, (x.shape, y.shape))
    # exit()
    model.save_weights('model_weight.hdf5')
    history = model.fit_generator(
        create_batch(8),
        steps_per_epoch=2000,
        epochs=50,
        verbose=1,
        class_weight=[0.2595, 0.1826, 4.5640, 0.1417, 0.5051, 0.3826, 9.6446, 1.8418, 6.6823, 6.2478, 3.0, 7.3614]
    )






if __name__ == '__main__':
    segnet = segnet.create_segnet((480, 360, 3))
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






