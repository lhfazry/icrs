## Indonesian Car Retrieval System

### Prerequisites

### Install Dependencies (including `Detectron2`)

```bash
pip install -r requirements.txt
```

#### Download Dataset

```bash
./run_download_dataset.sh
```

### A. Inference On Pretrained Model

#### 1. Download Pretrained Model

```bash
./run_download_pretrained_model.sh
```

#### 2. Run Inference

```bash
python inference.py --image_path path/to/image.jpg --model_path path/to/model.pth
```

### B.Training

#### 1. Training Detection Model

```bash
python train_detection.py
```

#### 2. Training Classification Model

```bash
python train_classification.py
```
