#!/bin/bash

###
# Small script to be executed on Centos to train my
# Deep Neural Network and upload the resulted model.h5 on Github.
# @author Celien Nanson
###

# Update packages dependencies
sudo yum -y update

# Install git, Python3.6 and git
sudo yum install git
sudo yum -y install yum-utils
sudo yum -y install https://centos7.iuscommunity.org/ius-release.rpm
sudo yum -y install python36u
sudo yum -y install python36u-pip

# Install model.py dependencies
sudo pip3.6 install numpy
sudo pip3.6 install sklearn
sudo pip3.6 install tensorflow-gpu
sudo pip3.6 install keras
