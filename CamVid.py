from typing import Tuple, List, Text, Dict, Any, Iterator, Union, Sized, Callable

import argparse
import time
import random
from datetime import datetime
import sys

sys.path.append("/usr/local/Cellar/opencv3/3.2.0/lib/python3.5/site-packages/") # mac opencv path
import cv2

import numpy as np
np.random.seed(2017) # for reproducibility

import os
#os.environ['KERAS_BACKEND'] = 'theano'
#os.environ["THEANO_FLAGS"] = "exception_verbosity=high,optimizer=None,device=cpu"
#os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=cpu,floatX=float32,optimizer=fast_compile'
os.environ['KERAS_BACKEND'] = 'tensorflow'
from keras.backend import set_image_data_format
# keras.backend.backend()
# keras.backend.set_epsilon(1e-07)
# keras.backend.epsilon()
# keras.backend.set_floatx('float32')
# keras.backend.floatx()
# set_image_data_format('channels_first') # theano
set_image_data_format("channels_last")
# keras.backend.image_data_format()
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint, Callback, TensorBoard
from keras.optimizers import SGD
import keras.backend.tensorflow_backend as KTF

import tensorflow as tf

from chainer import iterators

sys.path.append("./chainer-segnet")
from lib import CamVid

from SegNet import create_segnet


def train_iter_gen(train: Sized) -> Callable[[], Iterator[List[Tuple[np.ndarray, np.ndarray]]]] :
    return lambda: iterators.MultiprocessIterator(
        train,
        batch_size=8,
        n_processes=2,
        n_prefetch=2,
        #shared_mem=1024*1024*1024*4
    )

def valid_iter_gen(valid: Sized) -> Callable[[], Iterator[List[Tuple[np.ndarray, np.ndarray]]]] :
    return lambda: iterators.SerialIterator(
        valid,
        batch_size=16,
        #repeat=False,
        shuffle=False
    )

def convert_to_keras_batch(iter_gen: Callable[[], Iterator[List[Tuple[np.ndarray, np.ndarray]]]], n_classes: int, ignore_labels: List[int]) -> Iterator[Tuple[np.ndarray, np.ndarray]] :
    iter = iter_gen() # type: Iterator[List[Tuple[np.ndarray, np.ndarray]]]
    while True:
        batch = iter.__next__() # type: List[Tuple[np.ndarray, np.ndarray]]
        # len(batch) === 16
        # pair = batch[0] # type: Tuple[np.ndarray, np.ndarray]
        # img  = pair[0] # type: np.ndarray
        # mask = pair[1] # type: np.ndarray
        # img.shape == (3, 360, 480)
        # mask.shape == (360, 480)
        # str(img.dtype) == "float32"
        # str(mask.dtype) == "int32"
        xs = [np.einsum('chw->whc', x) for (x, _) in batch] # type: List[np.ndarray]

        ys = [] # type: List[np.ndarray]
        for (_, y) in batch:
            y = np.einsum('hw->wh', y)
            # https://github.com/pradyu1993/segnet/blob/master/segnet.py#L50
            (w, h) = y.shape # == (480, 360)
            _y = np.zeros((w, h, n_classes), dtype=np.uint8) # == (480, 360, n_classes)
            for i in range(w):
                for j in range(h):
                    _class = y[i][j]
                    if _class == -1: # ignore_labels
                        _class = ignore_labels[0]
                    _y[i, j, _class] = 1
            ys.append(_y)

        _xs = np.array(xs) # (n, 480, 360, 3)
        _ys = np.array(ys) # (n, 480, 360, n_classes)
        yield (_xs, _ys)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SegNet trainer from CamVid')
    parser.add_argument("--indices", action='store_true', help='use indices pooling')
    args = parser.parse_args()

    indices = args.indices

    class_weight = [float(w) for w in open("data/train_freq.csv").readline().split(',')] # type: List[float]
    n_classes = len(class_weight) # type: int
    ignore_labels = [11]

    train = CamVid(
        # https://github.com/pfnet-research/chainer-segnet/blob/master/lib/cmd_options.py
        img_dir="data/train",
        lbl_dir="data/trainannot",
        list_fn="data/train.txt",
        mean="data/train_mean.npy",
        std="data/train_std.npy",
        shift_jitter=50,
        scale_jitter=0.2,
        fliplr=True,
        rotate=True,
        rotate_max=7,
        scale=1.0,
        ignore_labels=ignore_labels,
    ) # type: Sized

    valid = CamVid(
        # https://github.com/pfnet-research/chainer-segnet/blob/master/lib/cmd_options.py
        img_dir="data/val",
        lbl_dir="data/valannot",
        list_fn="data/val.txt",
        mean="data/train_mean.npy",
        std="data/train_std.npy",
        ignore_labels=ignore_labels,
    ) # type: Sized

    train_iter = convert_to_keras_batch(train_iter_gen(train), n_classes, ignore_labels) # type: Iterator[Tuple[np.ndarray, np.ndarray]]
    valid_iter = convert_to_keras_batch(valid_iter_gen(valid), n_classes, ignore_labels) # type: Iterator[Tuple[np.ndarray, np.ndarray]]

    '''
    print(train_iter.__next__()[0].shape)
    print(train_iter.__next__()[0].shape)
    print(train_iter.__next__()[0].shape)
    print(train_iter.__next__()[0].shape)
    print(train_iter.__next__()[0].shape)

    exit()
    '''

    name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    if indices: name += "_indices"
    print("name: ", name)

    old_session = KTF.get_session()
    with tf.Graph().as_default():
        session = tf.Session("")
        KTF.set_session(session)
        KTF.set_learning_phase(1)

        segnet = create_segnet((480, 360, 3), n_classes, indices)
        segnet.compile(
            loss="categorical_crossentropy",
            optimizer=SGD(lr=0.01, momentum=0.8, decay=1e-6, nesterov=True),
            metrics=['accuracy']
        )

        with open(name+'_model.json', 'w') as f: f.write(segnet.to_json())
        segnet.save_weights(name+'_weight.hdf5')

        callbacks = [] # type: List[Callback]

        callbacks.append(ModelCheckpoint(
            name+"weights.{epoch:02d}-{val_loss:.2f}.hdf5",
            verbose=1,
            save_best_only=True,
            save_weights_only=True
        ))
        callbacks.append(TensorBoard(
            log_dir=name+'_log',
            histogram_freq=1,
            write_graph=True,
            write_images=True
        ))

        hist = segnet.fit_generator(
            generator=train_iter,
            steps_per_epoch=len(train),
            epochs=200,
            verbose=1,
            callbacks=callbacks,
            validation_data=valid_iter,
            validation_steps=len(valid),
            class_weight=class_weight,
            #initial_epoch
        )

        with open(name+'_history.json', 'w') as f: f.write(repr(hist.history))

    KTF.set_session(old_session)

    exit()


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
def batch_len(batch_size: int) -> int :
    with open('./SegNet-Tutorial/CamVid/train.txt', 'r') as f:
        return int(len(f.readlines())/batch_size)

def create_batch(batch_size: int=8, nb_class: int=12, ignored: int=11) -> Iterator[Tuple[np.ndarray, np.ndarray]] :
    return create_gen('./SegNet-Tutorial/CamVid/train.txt', batch_size, nb_class, ignored)

def create_valid(batch_size: int=8, nb_class: int=12, ignored: int=11) -> Iterator[Tuple[np.ndarray, np.ndarray]] :
    return create_gen('./SegNet-Tutorial/CamVid/test.txt', batch_size, nb_class, ignored)

def create_gen(filename: str, batch_size: int, nb_class: int, ignored: int) -> Iterator[Tuple[np.ndarray, np.ndarray]] :
    '''
    -> (float32, float32)
    '''
    with open(filename, 'r') as f:
        lines = f.readlines()
    pairs = [tuple(line.strip().replace('/SegNet', './SegNet-Tutorial').split(' ', 1)) for line in lines] # type: List[Tuple[str, str]]
    while True:
        to_shuffle = pairs[:]
        random.shuffle(to_shuffle)
        shuffled = iter(to_shuffle)
        while True:
            batch = [a for (a, b) in zip(shuffled, range(batch_size))]
            if len(batch) == 0: break
            loaded = [proc(preprocess_input(x), preprocess_teacher(y, nb_class, ignored)) for (x, y) in batch] # type: List[Tuple[np.ndarray, np.ndarray]]
            _x = np.array([x for (x, y) in loaded]) # (n, 480, 360, 3)
            _y = np.array([y for (x, y) in loaded]) # (n, 480, 360, nb_class)
            if _x.shape[0] != batch_size: break
            yield (_x, _y)

def proc(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray] :
    '''
    data augmentation
    uint8 -> float32
    ((w, h, 3), (w, h, 12))-> ((w, h, 3), (w, h, 12))
    '''
    if np.random.randint(0, 2) == 1:
        x = cv2.flip(x, 1)
        y = cv2.flip(y, 1)
    return (x.astype("float32"), y.astype("float32"))

def preprocess_input(filename: str) -> np.ndarray:
    '''
    -> uint8
    -> (w, h, 3)
    '''
    img = np.einsum('hwc->whc', cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB))
    _img = normalized(img)
    return _img

def preprocess_teacher(filename: str, nb_class: int, ignored: int) -> np.ndarray:
    '''
    labeling
    -> uint8
    -> (w, h, 12)
    '''
    # label は RGB すべてに 0~255 の範囲のクラス値が入っている
    img = np.einsum('hw->wh', cv2.imread(filename, cv2.IMREAD_GRAYSCALE))
    (w, h) = img.shape # == (480, 360)
    # https://github.com/pfnet-research/chainer-segnet/blob/ca84cd694351eeaff357656e76baa310dc455e66/lib/camvid.py#L63
    #img[np.where(img == ignored)] = -1 # ラベル ignored は Unlabelled
    _img = np.zeros((w, h, nb_class), dtype=np.uint8) # == (480, 360, nb_class)
    # https://github.com/pradyu1993/segnet/blob/master/segnet.py#L50
    for i in range(w):
        for j in range(h):
            _img[i, j, img[i][j]] = 1
    return _img

def normalized(rgb: np.ndarray) -> np.ndarray:
    '''
    equalizeHist for RGB
    return rgb/255.0
    uint8 -> uint8
    '''
    norm = np.ones(rgb.shape, rgb.dtype)*255

    r=rgb[:,:,0]
    g=rgb[:,:,1]
    b=rgb[:,:,2]

    norm[:,:,0]=cv2.equalizeHist(r)
    norm[:,:,1]=cv2.equalizeHist(g)
    norm[:,:,2]=cv2.equalizeHist(b)

    return norm



