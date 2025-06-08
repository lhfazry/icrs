#!/bin/bash

mkdir -p weights
gdown 1TdppuCurUb1oAMTCzGhApTny5Wb_IU0n -O weights/detection_best_model_config.pth
gdown 1nup0-qnc69FSUUgZSsRHmAH5N3dxCxXE -O weights/detection_best_model.pth

gdown 17sh59pI1NsNFXxwgLH45x5_3amV6jFS5 -O weights/classification_best_model.pth