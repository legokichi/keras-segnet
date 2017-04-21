
from typing import Tuple, List, Text, Dict, Any, Iterator, Union, Sized, Callable

import sys

sys.path.append("/usr/local/Cellar/opencv3/3.2.0/lib/python3.5/site-packages/") # mac opencv path
import cv2

import skimage.io as io
import numpy as np

from chainer.iterators import MultiprocessIterator, SerialIterator
from chainer.dataset import dataset_mixin

from imgaug import augmenters as iaa

sys.path.append("./coco/PythonAPI/")
from pycocotools.coco import COCO
from pycocotools import mask


class CamVid(dataset_mixin.DatasetMixin):
    def __init__(self, coco, resize_shape=None):
        self.resize_shape = resize_shape
        self.coco = coco
        self.infos = coco.loadImgs(coco.getImgIds(catIds=coco.getCatIds(catNms=['person']))) # type: List[dict]
    def __len__(self):
        return len(self.infos)
    def get_example(self, i):
        info = self.infos[i]
        img, mask = load_img(coco, info)
        img = np.expand_dims(img, axis=0)
        mask = np.expand_dims(mask, axis=0)
        img = seq.augment_images(img)
        mask = seq.augment_images(mask)
        img = np.squeeze(img)
        mask = np.squeeze(mask)
        if self.resize_shape != None:
            img = cv2.resize(img, resize_shape)
            mask = cv2.resize(mask, resize_shape)
        return (img, mask)

def load_img(coco, imgInfo: dict) -> Tuple[np.ndarray, np.ndarray]:
    img = io.imread(imgInfo['coco_url'])
    anns = coco.loadAnns(coco.getAnnIds(imgIds=[imgInfo['id']],iscrowd=False)) # type: List[dict]
    mask_all = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    for ann in anns:
        cat = coco.loadCats([ann["category_id"]])[0]
        if cat["name" ] != "person": continue
        rles = mask.frPyObjects(ann["segmentation"], img.shape[0], img.shape[1]) # type: List[dict]
        for i, rle in enumerate(rles):
            mask_img = mask.decode(rle) # type: np.ndarray
            mask_img[mask_img > 0] = 1
            mask_all += mask_img
    return (img, mask_all)


def convert_to_keras_batch(iter: Iterator[List[Tuple[np.ndarray, np.ndarray]]]) -> Iterator[Tuple[np.ndarray, np.ndarray]] :
    while True:
        batch = iter.__next__() # type: List[Tuple[np.ndarray, np.ndarray]]
        xs = [x for (x, _) in batch] # type: List[np.ndarray]
        ys = [y for (_, y) in batch] # type: List[np.ndarray]
        _xs = np.array(xs) # (n, 480, 360, 3)
        _ys = np.array(ys) # (n, 480, 360, n_classes)
        yield (_xs, _ys)


def get_iter(resize_shape=None)-> Tuple[Iterator[Tuple[np.ndarray, np.ndarray]], Iterator[Tuple[np.ndarray, np.ndarray]]]:

    coco = COCO("./annotations/instances_train2014.json")

    seq = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
        iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                translate_px={"x": (-16, 16), "y": (-16, 16)}, # translate by -16 to +16 pixels (per axis)
                rotate=(-45, 45), # rotate by -45 to +45 degrees
                shear=(-16, 16), # shear by -16 to +16 degrees
                #order=iaa.ALL, # use any of scikit-image's interpolation methods
                #cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                mode="wrap" # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        ),
    ]).to_deterministic() # type: iaa.Sequential

    train_iter = convert_to_keras_batch(
        #SerialIterator(
        MultiprocessIterator(
            CamVid(coco, resize_shape),
            batch_size=8,
            n_processes=4,
            n_prefetch=4,
            shared_mem=1000*1000*5
        )
    )


    valid_iter = convert_to_keras_batch(
        #SerialIterator(
        MultiprocessIterator(
            CamVid(coco, resize_shape),
            batch_size=8,
            #repeat=False,
            shuffle=False,
            n_processes=4,
            n_prefetch=4,
            shared_mem=1000*1000*5
        )
    ) # type: Iterator[Tuple[np.ndarray, np.ndarray]]

    return train_iter, valid_iter