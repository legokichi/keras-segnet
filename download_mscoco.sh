#pyenv shell anaconda3-4.2.0
echo 'export PYTHONPATH=`pwd`/coco/PythonAPI:$PYTHONPATH' >> /home/ubuntu/.zshrc
cd coco/PythonAPI; make
cd ../../
wget http://msvocds.blob.core.windows.net/annotations-1-0-3/instances_train-val2014.zip
unzip instances_train-val2014.zip
\rm instances_train-val2014.zip
# python -c 'from pycocotools.coco import COCO; coco = COCO("./annotations/instances_train2014.json")'

