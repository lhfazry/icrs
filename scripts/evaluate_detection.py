import os
import torch

from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from scripts.train_detection import register_datasets
from detectron2.config import get_cfg
from utils.path_util import ensure_root
from config import DATASET_NAME, DETECTION_OUTPUT_DIR, DETECTION_EVAL_OUTPUT_DIR, DETECTION_THRESHOLD

def evaluate():
    register_datasets()
    
    cfg = torch.load(os.path.join(DETECTION_OUTPUT_DIR, "detection_best_model_config.pth"), weights_only=False)
    cfg.MODEL.WEIGHTS = os.path.join(DETECTION_OUTPUT_DIR, "detection_best_model.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = DETECTION_THRESHOLD
    predictor = DefaultPredictor(cfg)
    
    evaluator = COCOEvaluator(f"{DATASET_NAME}_test", output_dir=DETECTION_EVAL_OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, f"{DATASET_NAME}_test")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))

if __name__ == "__main__":
    ensure_root()
    evaluate()