#!/bin/bash

# Update pip and setuptools first
pip install --upgrade pip setuptools

# Install pyyaml with newer version
pip install pyyaml==5.4.1

# Remove existing detectron2 directory if exists
rm -rf detectron2

# Clone detectron2 repository
git clone 'https://github.com/facebookresearch/detectron2'

# Install detectron2
cd detectron2
pip install -e .
cd ..

# Add detectron2 to Python path
export PYTHONPATH=$(pwd)/detectron2:$PYTHONPATH