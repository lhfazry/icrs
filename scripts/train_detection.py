import os
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger
from utils.trainer import Trainer
from utils.path_util import ensure_root
from config import DATASET_NAME, DETECTION_OUTPUT_DIR, NUM_CLASSES

# Setup logger
setup_logger()

# Dataset registration
def register_datasets():
    register_coco_instances(f"{DATASET_NAME}_train", {}, "data/detection/train/_annotations.coco.json", "data/detection/train")
    register_coco_instances(f"{DATASET_NAME}_val", {}, "data/detection/valid/_annotations.coco.json", "data/detection/valid")
    register_coco_instances(f"{DATASET_NAME}_test", {}, "data/detection/test/_annotations.coco.json", "data/detection/test")

def train_model():
    # Register datasets
    register_datasets()
    
    # Configuration
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = (f"{DATASET_NAME}_train",)
    cfg.DATASETS.TEST = (f"{DATASET_NAME}_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 500
    cfg.SOLVER.STEPS = []
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # Use GPU if available
    
    # Output directory
    os.makedirs(DETECTION_OUTPUT_DIR, exist_ok=True)
    cfg.OUTPUT_DIR = DETECTION_OUTPUT_DIR
    
    # Train
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    model_name = "detection_best_model"
    # Save final weights with custom name
    final_model_path = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    custom_model_path = os.path.join(cfg.OUTPUT_DIR, f"{model_name}.pth")
    
    # Rename the default final model to your custom name
    if os.path.exists(final_model_path):
        os.rename(final_model_path, custom_model_path)
    
    # Also save config with matching name
    torch.save(cfg, os.path.join(cfg.OUTPUT_DIR, f"{model_name}_config.pth"))
    
    print(f"Training completed! Model saved as {custom_model_path}")

if __name__ == "__main__":
    ensure_root()
    
    train_model()