#!/bin/bash

###
# Small script to be executed on Centos to train my
# Deep Neural Network and upload the resulted model.h5 on Github.
# @author Celien Nanson
###

# Install sudo
yum -y install sudo

# Update packages dependencies
sudo yum -y update

# Install git, Python3.6, git and wget
sudo yum -y install git
sudo yum -y install yum-utils
sudo yum -y install https://centos7.iuscommunity.org/ius-release.rpm
sudo yum -y install python36u
sudo yum -y install python36u-pip
sudo yum -y install wget

# Install model.py dependencies
sudo pip3.6 install numpy
sudo pip3.6 install sklearn
sudo pip3.6 install tensorflow
sudo pip3.6 install keras
sudo pip3.6 install python-mnist

# Download MNIST dataset and decompress it
mkdir data/ && cd data/
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
gzip -d train-images-idx3-ubyte.gz
gzip -d train-labels-idx1-ubyte.gz
cd ..

# Train the model
python3.6 model.py
