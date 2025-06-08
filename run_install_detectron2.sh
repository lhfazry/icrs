#!/bin/bash

# Pertama, pastikan environment bersih
pip uninstall -y pyyaml detectron2 fvcore

# Install dependencies system-level (untuk Colab/Ubuntu)
sudo apt-get update
sudo apt-get install -y build-essential python3-dev libopenmpi-dev

# Buat virtual environment (opsional tapi direkomendasikan)
python -m venv venv
source venv/bin/activate

# Update pip dan setuptools dengan cara yang lebih aman
python -m pip install --upgrade "pip<25" "setuptools<80"

# Install PyYAML dengan cara khusus
pip install --no-cache-dir --force-reinstall -Iv pyyaml==5.4.1

# Clone detectron2 (pastikan direktori kosong)
rm -rf detectron2
git clone https://github.com/facebookresearch/detectron2.git

# Install dengan opsi khusus
cd detectron2
pip install -e . \
  --no-build-isolation \
  --config-settings editable_mode=compat \
  --use-pep517
cd ..

# Verifikasi instalasi
python -c "import detectron2; print(detectron2.__version__)"