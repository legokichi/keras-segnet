#pyenv shell anaconda3-4.2.0
echo 'export PYTHONPATH=`pwd`/pascal-voc-python:$PYTHONPATH' >> /home/ubuntu/.zshrc
cd pascal-voc-python; python setup.py install
cd ../
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_11-May-2012.tar
\rm VOCtrainval_11-May-2012.tar
# python -c 'import voc_utils'

