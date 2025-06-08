import os
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn as nn
from utils.transform_util import get_transforms
from torchvision import models
from utils.car_dataset import CarDataset
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from utils.path_util import ensure_root
from config import CLASSIFICATION_DATASET_ROOT, CLASSIFICATION_BATCH_SIZE, \
    CLASSIFICATION_MODEL_TYPE, CLASSIFICATION_OUTPUT_DIR, \
    CLASSIFICATION_EVAL_OUTPUT_DIR, DEVICE, NUM_CLASSES

MODEL_TYPE = CLASSIFICATION_MODEL_TYPE

def evaluate_model():
    # Load best model
    model = models.__dict__[MODEL_TYPE](pretrained=False)
    
    if MODEL_TYPE.startswith("resnet"):
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    elif MODEL_TYPE.startswith("efficientnet"):
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, NUM_CLASSES)
    
    model.load_state_dict(torch.load(os.path.join(CLASSIFICATION_OUTPUT_DIR, "classification_best_model.pth")))
    model = model.to(DEVICE)
    model.eval()
    
    # Load test data
    _, test_transform = get_transforms()
    test_dataset = CarDataset(os.path.join(CLASSIFICATION_DATASET_ROOT, "test"), test_transform)
    test_loader = DataLoader(test_dataset, batch_size=CLASSIFICATION_BATCH_SIZE, shuffle=False)
    
    # Evaluation
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Generate classification report
    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=test_dataset.classes, 
                yticklabels=test_dataset.classes)
    plt.title("Confusion Matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(os.path.join(CLASSIFICATION_EVAL_OUTPUT_DIR, "confusion_matrix.png"))
    plt.close()

if __name__ == "__main__":
    ensure_root()
    evaluate_model()