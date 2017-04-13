bash chainer-segnet/experiments/download.sh 
export PYTHONPATH=/usr/local/Cellar/opencv3/3.2.0/lib/python3.5/site-packages/:$PYTHONPATH
export PYTHONPATH=`pwd`/chainer-segnet:$PYTHONPATH
python chainer-segnet/lib/calc_mean.py
