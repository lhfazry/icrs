import torch
import os
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.car_dataset import CarDataset
from utils.transform_util import get_transforms
from tqdm import tqdm
from utils.path_util import ensure_root
from config import CLASSIFICATION_DATASET_ROOT, CLASSIFICATION_BATCH_SIZE, \
    CLASSIFICATION_NUM_EPOCHS, CLASSIFICATION_LEARNING_RATE, \
    CLASSIFICATION_NUM_WORKERS, CLASSIFICATION_MODEL_TYPE, \
    CLASSIFICATION_OUTPUT_DIR

# Configuration
DATASET_ROOT = CLASSIFICATION_DATASET_ROOT
BATCH_SIZE = CLASSIFICATION_BATCH_SIZE
NUM_EPOCHS = CLASSIFICATION_NUM_EPOCHS
LEARNING_RATE = CLASSIFICATION_LEARNING_RATE
NUM_WORKERS = CLASSIFICATION_NUM_WORKERS
MODEL_TYPE = CLASSIFICATION_MODEL_TYPE
PRETRAINED = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = CLASSIFICATION_OUTPUT_DIR

def train_model():
    writer = SummaryWriter(log_dir=OUTPUT_DIR)
    
    train_transform, val_transform = get_transforms()
    
    train_dataset = CarDataset(os.path.join(DATASET_ROOT, "train"), train_transform)
    val_dataset = CarDataset(os.path.join(DATASET_ROOT, "valid"), val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                             shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                           shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    if MODEL_TYPE.startswith("resnet"):
        model = models.__dict__[MODEL_TYPE](pretrained=PRETRAINED)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))
    elif MODEL_TYPE.startswith("efficientnet"):
        model = models.__dict__[MODEL_TYPE](pretrained=PRETRAINED)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, len(train_dataset.classes))
    
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    best_acc = 0.0
    
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print("-" * 10)
        
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        writer.add_scalar("Accuracy/train", epoch_acc, epoch)
        
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in tqdm(val_loader, desc="Validation"):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            with torch.set_grad_enabled(False):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_loss = running_loss / len(val_dataset)
        epoch_acc = running_corrects.double() / len(val_dataset)
        
        writer.add_scalar("Loss/val", epoch_loss, epoch)
        writer.add_scalar("Accuracy/val", epoch_acc, epoch)
        
        print(f"Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        
        # Save best model
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "classification_best_model.pth"))
        
        scheduler.step()
    
    writer.close()
    print(f"Training complete. Best val Acc: {best_acc:.4f}")

if __name__ == "__main__":
    ensure_root()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_model()