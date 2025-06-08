#!/bin/bash

gdown 1I6WY21iCs_XR8wFWM-79JRryp_9PunMT
mv input_video.mp4 data

gdown 1V3jDKzgbIKk1fLTpYWScM9O2h2o9pPh1
mkdir -p data/detection
unzip icrs.zip -d data/detection > /dev/null