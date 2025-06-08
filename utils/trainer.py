import os
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs("./output/detection_train_eval_output", exist_ok=True)
            output_folder = "./output/detection_train_eval_output"
        return COCOEvaluator(dataset_name, cfg, False, output_folder)