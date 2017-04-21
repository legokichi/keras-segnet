from typing import Tuple, List, Text, Dict, Any, Iterator, Union, Sized, Callable

import cProfile
import pstats

import time
import random
from datetime import datetime
import sys

sys.path.append("/usr/local/Cellar/opencv3/3.2.0/lib/python3.5/site-packages/") # mac opencv path
import cv2

import numpy as np

from chainer.iterators import MultiprocessIterator, SerialIterator

sys.path.append("./chainer-segnet")
from lib import CamVid

class _CamVid(CamVid):

    def __init__(self, n_classes: int, resize_shape=None, *args, **kwargs):
        self.n_classes = n_classes
        self.resize_shape = resize_shape
        super().__init__(*args, **kwargs)

    def get_example(self, i) -> Tuple[np.ndarray, np.ndarray] :
        ret = CamVid.get_example(self, i) # type: Tuple[np.ndarray, np.ndarray]
        (x, y) = ret
        assert x.shape == (3, 360, 480)
        assert y.shape == (360, 480)
        assert str(x.dtype) == "float32"
        assert str(y.dtype) == "int32"
        _x = np.einsum('chw->whc', x)
        y = np.einsum('hw->wh', y)
        # https://github.com/pradyu1993/segnet/blob/master/segnet.py#L50
        (w, h) = y.shape
        _y = np.zeros((w, h, self.n_classes), dtype=np.uint8) # == (480, 360, n_classes)
        for i in range(self.n_classes):
            _y[:,:,i] = (y == i).astype("uint8")
        if len(self.ignore_labels) > 0:
            _y[:,:,self.ignore_labels[0]] = (y == -1).astype("uint8") # ignore_labels
        assert _x.shape == (480, 360, 3)
        assert _y.shape == (480, 360, 12)
        assert str(_x.dtype) == "float32"
        assert str(_y.dtype) == "uint8"
        if self.resize_shape != None:
            _x = cv2.resize(_x, self.resize_shape)
            _y = cv2.resize(_y, self.resize_shape)
        return (_x, _y)


def convert_to_keras_batch(iter: Iterator[List[Tuple[np.ndarray, np.ndarray]]]) -> Iterator[Tuple[np.ndarray, np.ndarray]] :
    #batch_size = 8
    #n_classes = 12
    while True:
        batch = iter.__next__() # type: List[Tuple[np.ndarray, np.ndarray]]
        #assert len(batch) == batch_size
        xs = [x for (x, _) in batch] # type: List[np.ndarray]
        ys = [y for (_, y) in batch] # type: List[np.ndarray]
        _xs = np.array(xs) # (n, 480, 360, 3)
        _ys = np.array(ys) # (n, 480, 360, n_classes)
        #assert _xs.shape == (batch_size, 480, 360, 3)
        #assert _ys.shape == (batch_size, 480, 360, n_classes)
        yield (_xs, _ys)


def get_iter(resize_shape=None)-> Tuple[Iterator[Tuple[np.ndarray, np.ndarray]], Iterator[Tuple[np.ndarray, np.ndarray]]]:

    class_weight = [float(w) for w in open("data/train_freq.csv").readline().split(',')] # type: List[float]
    n_classes = len(class_weight) # type: int
    ignore_labels = [11]
    
    train = _CamVid(
        n_classes=n_classes,
        resize_shape=resize_shape,
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

    valid = _CamVid(
        n_classes=n_classes,
        resize_shape=resize_shape,
        # https://github.com/pfnet-research/chainer-segnet/blob/master/lib/cmd_options.py
        img_dir="data/val",
        lbl_dir="data/valannot",
        list_fn="data/val.txt",
        mean="data/train_mean.npy",
        std="data/train_std.npy",
        ignore_labels=ignore_labels,
    ) # type: Sized
    
    train_iter = convert_to_keras_batch(
        #SerialIterator(
        MultiprocessIterator(
            train,
            batch_size=8,
            n_processes=2,
            n_prefetch=2,
            shared_mem=1000*1000*5
        )
    ) # type: Iterator[Tuple[np.ndarray, np.ndarray]]

    valid_iter = convert_to_keras_batch(
        #SerialIterator(
        MultiprocessIterator(
            valid,
            batch_size=8,
            #repeat=False,
            shuffle=False,
            n_processes=2,
            n_prefetch=2,
            shared_mem=1000*1000*5
        )
    ) # type: Iterator[Tuple[np.ndarray, np.ndarray]]

    return (train_iter, valid_iter)

if __name__ == '__main__':
    train_iter = get_iter()
    def main():
        start = time.time()
        for i in range(100):
            train_iter.__next__()
            time.sleep(1)
        end = time.time()
        print('%30s' % 'serial in ', str((end - start)*1000), 'ms')

    cProfile.run("main()", filename="main.prof")
    sts = pstats.Stats("main.prof")
    sts.strip_dirs().sort_stats("cumulative").print_stats()

    exit()