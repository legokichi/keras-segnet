# keras-segnet

working in progress

## ref

* https://github.com/pradyu1993/segnet/
* https://github.com/pradyu1993/segnet/issues/10
* https://github.com/imlab-uiip/keras-segnet
* https://github.com/yandex/segnet-torch
* https://gist.github.com/rdelassus/5b908efd07ae030a2650584e199ff25b/
* https://github.com/alexgkendall/caffe-segnet
* https://github.com/alexgkendall/SegNet-Tutorial
* https://github.com/alexfailure/SegNet/

## install

```
git clone --recursive https://github.com/legokichi/keras-segnet.git
pyenv shell anaconda3-4.1.1
sudo apt-get install graphviz
conda install theano pygpu
pip install tensorflow-gpu
pip install keras
pip install mypy
pip install pydot_ng
pip install imgaug

```

## type check

```
mypy --ignore-missing-imports train.py 
```

## show model

```
python model_segnet.py
python model_unet.py
```

## train

```
source download_camvid.sh
source download_mscoco.sh
env CUDA_VISIBLE_DEVICES=0 python train.py
env CUDA_VISIBLE_DEVICES=1 python train.py --indices
env CUDA_VISIBLE_DEVICES=0 python train.py --unet --coco
env CUDA_VISIBLE_DEVICES=0 python train.py --unet --coco --ker_init=he_normal --lr=0.001 --optimizer=nesterov --loss=dice_coef
tensorboard --port=8888 --logdir=log
jupyter notebook --ip=0.0.0.0
```

If learning does not start, memory is not enough so please change MultiprocessIterator to SerialIterator in CamVid.py or mscoco.py.

### resume

```
env CUDA_VISIBLE_DEVICES=0 python train.py --initial_epoch=5 --resume=2017-04-17-08-29-19_weights.epoch0005-val_loss0.43.hdf5 
env CUDA_VISIBLE_DEVICES=1 python train.py --initial_epoch=5 --resume=2017-04-17-08-31-47_indices_weights.epoch0005-val_loss0.48.hdf5 --indices
```

## predict

working in progress

## model
### segnet

![segnet](https://raw.githubusercontent.com/legokichi/keras-segnet/master/segnet.png)

### u-net

![u-net](https://raw.githubusercontent.com/legokichi/keras-segnet/master/unet.png)
