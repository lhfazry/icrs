import torch

## General
DATASET_NAME = 'icrs'
NUM_CLASSES = 7
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHT_FOLDER = "weights"

## Detection
DETECTION_THRESHOLD = 0.7
DETECTION_OUTPUT_DIR = "./output/detection_train"
DETECTION_EVAL_OUTPUT_DIR = "./output/detection_eval"
DETECTION_DATASET_ROOT = "data/detection"

## Classification
CLASSIFICATION_THRESHOLD = 0.5
CLASSIFICATION_OUTPUT_DIR = "./output/classification_train"
CLASSIFICATION_EVAL_OUTPUT_DIR = "./output/classification_eval"
CLASSIFICATION_DATASET_ROOT = "data/classification"
CLASSIFICATION_BATCH_SIZE = 32
CLASSIFICATION_NUM_EPOCHS = 50
CLASSIFICATION_LEARNING_RATE = 0.001
CLASSIFICATION_NUM_WORKERS = 4
CLASSIFICATION_MODEL_TYPE = "efficientnet_b0"