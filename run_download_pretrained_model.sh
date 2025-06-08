#!/bin/bash

mkdirs -p weights
gdown 1V3jDKzgbIKk1fLTpYWScM9O2h2o9pPh1 -O weights/classification_best_model.pth
gdown 1V3jDKzgbIKk1fLTpYWScM9O2h2o9pPh1 -O weights/detection_best_model.pth