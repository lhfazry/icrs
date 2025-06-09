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

The dataset will be saved on `data/detection` folder. This is a detection dataset in COCO format. For classification dataset, we can convert it into classification dataset using `./run_dataset_conversion.sh`. This script also download the input video file into `data/input_video.mp4`.

```bash
./run_dataset_conversion.sh
```

The dataset will be saved on `data/classification` folder. To explore the dataset, you can use provided notebook [ICRS Dataset Exploration](notebooks/ICRS_Dataset_Exploration.ipynb).

### A. Inference On Pretrained Model

#### 1. Download Pretrained Model

```bash
./run_download_pretrained_model.sh
```

The weights will be saved on `weights` folder.

#### 2. Run Inference

```bash
./run_inference.sh
```

The inference script assume there is a video file located at `data/input_video.mp4` and the result will be saved on `data/output_video.mp4`.

### B.Training

#### 1. Training Detection Model

```bash
./run_train_detection.sh
```

#### 2. Training Classification Model

```bash
./run_train_classification.sh
```
