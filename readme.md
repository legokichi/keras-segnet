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

## install

```
pyenv shell anaconda3-4.1.1
brew install graphviz
conda install theano pygpu
pip install tensorflow-gpu
pip install keras
pip install mypy
pip install pydot_ng
```

## type check

```
mypy --ignore-missing-imports CamVid.py 
```

## show model

```
python SegNet.py
```

## train

```
env CUDA_VISIBLE_DEVICES=0 python CamVid.py
env CUDA_VISIBLE_DEVICES=1 python CamVid.py --indices
```

## predict

working in progress

## model

![segnet](https://raw.githubusercontent.com/legokichi/keras-segnet/master/segnet.png)


