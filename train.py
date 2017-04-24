from typing import Tuple, List, Text, Dict, Any, Iterator, Union, Sized, Callable

import cProfile
import pstats
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

from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint, Callback, TensorBoard
from keras.optimizers import SGD, Adam
from keras.backend import tensorflow_backend
import keras.backend as K
import tensorflow as tf

from chainer.iterators import MultiprocessIterator, SerialIterator

sys.path.append("./chainer-segnet")
from lib import CamVid

from SegNet import create_segnet
from unet import create_unet
from CamVid import get_iter as get_camvid
from mscoco import get_iter as get_coco

def convert_to_keras_batch(iter: Iterator[List[Tuple[np.ndarray, np.ndarray]]]) -> Iterator[Tuple[np.ndarray, np.ndarray]] :
    while True:
        batch = iter.__next__() # type: List[Tuple[np.ndarray, np.ndarray]]
        xs = [x for (x, _) in batch] # type: List[np.ndarray]
        ys = [y for (_, y) in batch] # type: List[np.ndarray]
        _xs = np.array(xs) # (n, 480, 360, 3)
        _ys = np.array(ys) # (n, 480, 360, n_classes)
        yield (_xs, _ys)



def dice_coef(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    return (2. * intersection + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SegNet trainer from CamVid')
    parser.add_argument("--indices", action='store_true', help='use indices pooling')
    parser.add_argument("--epochs",  action='store', type=int, default=1000, help='epochs')
    parser.add_argument("--resume",  action='store', type=str, default="", help='*_weights.hdf5')
    parser.add_argument("--initial_epoch", action='store', type=int, default=0, help='initial_epoch')
    parser.add_argument("--unet", action='store_true', help='use u-net')
    parser.add_argument("--coco", action='store_true', help='use mscoco dataset')
    parser.add_argument("--ker_init", action='store', type=str, default="glorot_uniform", help='conv2D kernel initializer')
    parser.add_argument("--lr", action='store', type=float, default=0.001, help='learning late')
    parser.add_argument("--optimizer", action='store', type=str, default="adam", help='adam|nesterov')
    parser.add_argument("--loss", action='store', type=str, default="categorical_crossentropy", help='dice_coef|categorical_crossentropy')
    
    args = parser.parse_args()

    if args.unet: resize_shape = (512, 512)
    else: resize_shape = None

    if args.coco:
      train, valid = get_coco(resize_shape)
    else:
      train, valid = get_camvid(resize_shape)

    train_iter = convert_to_keras_batch(
        #SerialIterator(
        MultiprocessIterator(
            train,
            batch_size=8,
            n_processes=12,
            n_prefetch=120,
            shared_mem=1000*1000*5
        )
    )


    valid_iter = convert_to_keras_batch(
        #SerialIterator(
        MultiprocessIterator(
            valid,
            batch_size=8,
            #repeat=False,
            shuffle=False,
            n_processes=12,
            n_prefetch=120,
            shared_mem=1000*1000*5
        )
    ) # type: Iterator[Tuple[np.ndarray, np.ndarray]]
    
    name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    if args.unet: name += "_unet"
    elif args.indices: name += "_indices"
    if args.coco: name += "_coco"
    
    name += "_" + args.loss
    name += "_" + args.optimizer
    name += "_lr" + str(args.lr)
    name += "_" + args.ker_init

    print("name: ", name)

    old_session = tensorflow_backend.get_session()

    with tf.Graph().as_default():
        session = tf.Session("")
        tensorflow_backend.set_session(session)
        tensorflow_backend.set_learning_phase(1)

        if args.unet:
            loss_weights=None
            segnet = create_unet((512, 512, 3), (512, 512, 2), 128, args.ker_init)
        else:
            loss_weights = None #[0.2595, 0.1826, 4.5640, 0.1417, 0.9051, 0.3826, 9.6446, 1.8418, 0.6823, 6.2478, 7.3614], # https://github.com/alexgkendall/SegNet-Tutorial/blob/master/Models/bayesian_segnet_train.prototxt#L1615
            n_classes = 12
            segnet = create_segnet((480, 360, 3), n_classes, args.indices, args.ker_init)
        
        if args.optimizer == "nesterov": optimizer = SGD(lr=args.lr, momentum=0.9, decay=0.0005, nesterov=True)
        else: optimizer = Adam(lr=args.lr, beta_1=0.5, beta_2=0.999, epsilon=1e-08, decay=0.0)
        if args.loss == "dice_coef_loss":
            loss = dice_coef_loss
            metrics = ['dice_coef']
        else:
            loss = "categorical_crossentropy"
            metrics = ['accuracy']

        segnet.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            loss_weights=loss_weights
        )
        if len(args.resume) > 0:
            segnet.load_weights(args.resume)

        with open(name+'_model.json', 'w') as f: f.write(segnet.to_json())

        callbacks = [] # type: List[Callback]

        callbacks.append(ModelCheckpoint(
            name+"_weights.epoch{epoch:04d}-val_loss{val_loss:.2f}.hdf5",
            verbose=1,
            #save_best_only=True,
            save_weights_only=True,
            period=1,
        ))

        callbacks.append(TensorBoard(
            log_dir=name+'_log',
            histogram_freq=1,
            write_graph=False,
            write_images=False,
        ))

        hist = segnet.fit_generator(
            generator=train_iter,
            steps_per_epoch=len(train),
            epochs=args.epochs,
            verbose=1,
            callbacks=callbacks,
            validation_data=valid_iter,
            validation_steps=2,
            #class_weight=class_weight,
            initial_epoch=args.initial_epoch,
        )

        with open(name+'_history.json', 'w') as f: f.write(repr(hist.history))
        segnet.save_weights(name+'_weight_final.hdf5')

    tensorflow_backend.set_session(old_session)

    print("entering repl (loop=False to exit)")
    global loop
    loop = True
    while loop:
        try:
            exec(input("> "))
        except:
            print("Unexpected error:", sys.exc_info()[0])
    print("finish")



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



