
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
    def __init__(self, coco, path, seq=None, resize_shape=None):
        self.resize_shape = resize_shape
        self.coco = coco
        self.infos = coco.loadImgs(coco.getImgIds(catIds=coco.getCatIds(catNms=['person']))) # type: List[dict]
        self.seq = seq
        self.seq_norm = iaa.Sequential([
            iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5)
        ])
        self.path = path
    def __len__(self):
        return len(self.infos)
    def get_example(self, i):
        info = self.infos[i]
        img, mask = load_img(self.coco, self.path, info)
        if self.seq != None:
            img = np.expand_dims(img, axis=0)
            mask = np.expand_dims(mask, axis=0)
            img = self.seq.augment_images(img)
            img = self.seq_norm.augment_images(img)
            mask = self.seq.augment_images(mask)
            img = np.squeeze(img)
            mask = np.squeeze(mask)
        if self.resize_shape != None:
            img = cv2.resize(img, self.resize_shape)
            mask = cv2.resize(mask, self.resize_shape)
        mask[:,:,0] = mask[:,:,0]>0
        return (img, mask)

def load_img(coco, path: str, imgInfo: dict) -> Tuple[np.ndarray, np.ndarray]:
    img = io.imread(path +imgInfo['file_name'])
    if img.ndim != 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    anns = coco.loadAnns(coco.getAnnIds(imgIds=[imgInfo['id']],iscrowd=False)) # type: List[dict]
    mask_all = np.zeros((img.shape[0], img.shape[1], 2), np.uint8)
    for ann in anns:
        cat = coco.loadCats([ann["category_id"]])[0]
        if cat["name" ] != "person": continue
        rles = mask.frPyObjects(ann["segmentation"], img.shape[0], img.shape[1]) # type: List[dict]
        for i, rle in enumerate(rles):
            mask_img = mask.decode(rle) # type: np.ndarray
            mask_all[:,:,0] += mask_img
    return (img, mask_all)



def get_iter(resize_shape=None):

    coco_train = COCO("./annotations/instances_train2014.json")
    coco_val = COCO("./annotations/instances_val2014.json")

    seq = iaa.Sequential([
        iaa.Fliplr(0.5),
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

    return CamVid(coco_train, "train2014/", seq, resize_shape), CamVid(coco_val, "val2014/", resize_shape)
