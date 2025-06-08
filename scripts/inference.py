import cv2
import torch
import numpy as np
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
from utils.path_util import ensure_root
from torchvision import models, transforms

# Configuration
WEIGHT_FOLDER = "weights"
VIDEO_PATH = "input_video.mp4"
OUTPUT_PATH = "output_video.mp4"
DETECTION_THRESHOLD = 0.7
CLASSIFICATION_THRESHOLD = 0.5
CAR_CLASS_ID = 2  # COCO class ID for cars
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 7

# 1. Initialize Detection Model (Detectron2)
def setup_detector():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = DETECTION_THRESHOLD
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.DEVICE = DEVICE
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = NUM_CLASSES
    return DefaultPredictor(cfg)

# 2. Initialize Classification Model (ResNet/EfficientNet)
def setup_classifier(model_type="efficientnet"):
    num_classes = 7
    
    if model_type == "resnet":
        model = models.resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(torch.load(os.path.join(WEIGHT_FOLDER, "resnet_car_model.pth")))
    elif model_type == "efficientnet":
        model = models.efficientnet_b0(pretrained=False)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
        model.load_state_dict(torch.load(os.path.join(WEIGHT_FOLDER, "efficientnet_car_model.pth")))
    
    model = model.to(DEVICE)
    model.eval()
    return model

# 3. Preprocessing for classification
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(Image.fromarray(image)).unsqueeze(0).to(DEVICE)

# 4. Load class names for classification
def load_class_names(path="car_classes.txt"):
    with open(path) as f:
        return [line.strip() for line in f.readlines()]

# Main processing pipeline
def process_video():
    # Initialize models
    detector = setup_detector()
    classifier = setup_classifier("efficientnet")  # or "resnet"
    class_names = load_class_names()
    
    # Video setup
    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
    
    # Process each frame
    for _ in tqdm(range(total_frames), desc="Processing video"):
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detection
        outputs = detector(frame_rgb)
        instances = outputs["instances"]
        
        # Filter only cars
        car_indices = (instances.pred_classes == CAR_CLASS_ID).nonzero().flatten()
        car_boxes = instances.pred_boxes[car_indices]
        car_scores = instances.scores[car_indices]
        
        # Process each detected car
        for box, score in zip(car_boxes, car_scores):
            x1, y1, x2, y2 = box.cpu().numpy().astype(int)
            
            # Crop and classify
            car_crop = frame_rgb[y1:y2, x1:x2]
            if car_crop.size == 0:
                continue
                
            # Preprocess and classify
            input_tensor = preprocess_image(car_crop)
            with torch.no_grad():
                output = classifier(input_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                max_prob, pred_class = torch.max(probabilities, 1)
                
            if max_prob.item() > CLASSIFICATION_THRESHOLD:
                car_type = class_names[pred_class.item()]
                label = f"{car_type} ({max_prob.item():.2f})"
                
                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Write frame to output
        out.write(frame)
    
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    ensure_root()
    process_video()