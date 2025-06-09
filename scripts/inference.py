import cv2
import torch, os
from detectron2.engine import DefaultPredictor
from torchvision import models
from PIL import Image
from tqdm import tqdm
from utils.path_util import ensure_root
from config import DEVICE, DETECTION_THRESHOLD, CLASSIFICATION_THRESHOLD
from utils.transform_util import get_transforms

# Configuration
WEIGHT_FOLDER = "weights"
VIDEO_PATH = "data/input_video.mp4"
OUTPUT_PATH = "output/output_video.mp4"
class_names = ['Bajaj', 'Box', 'Bus', 'Hatchback', 'MPV_SUV', 'Minibus', 'Pickup-Truck', 'Sedan']

def setup_detector():
    cfg = torch.load(os.path.join(WEIGHT_FOLDER, "detection_best_model_config.pth"), weights_only=False)
    cfg.MODEL.WEIGHTS = os.path.join(WEIGHT_FOLDER, "detection_best_model.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = DETECTION_THRESHOLD
    return DefaultPredictor(cfg)

def setup_classifier(model_type="efficientnet"):
    num_classes = len(class_names)
    
    if model_type == "resnet":
        model = models.resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(torch.load(os.path.join(WEIGHT_FOLDER, "classification_best_model.pth")))
    elif model_type == "efficientnet":
        model = models.efficientnet_b0(pretrained=False)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
        model.load_state_dict(torch.load(os.path.join(WEIGHT_FOLDER, "classification_best_model.pth")))
    
    model = model.to(DEVICE)
    model.eval()
    return model

def preprocess_image(image):
    _, test_transform = get_transforms()
    return test_transform(Image.fromarray(image)).unsqueeze(0).to(DEVICE)

def process_video():
    detector = setup_detector()
    classifier = setup_classifier("efficientnet")  # or "resnet"
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
    
    # Process each frame
    for _ in tqdm(range(total_frames), desc="Processing video"):
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        outputs = detector(frame_rgb)
        instances = outputs["instances"]
        
        car_boxes = instances.pred_boxes
        car_scores = instances.scores
        
        for box, _ in zip(car_boxes, car_scores):
            x1, y1, x2, y2 = box.cpu().numpy().astype(int)
            
            car_crop = frame_rgb[y1:y2, x1:x2]
            if car_crop.size == 0:
                continue
                
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
        
        out.write(frame)
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    ensure_root()
    os.makedirs("output", exist_ok=True)
    process_video()