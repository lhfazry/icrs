#!/bin/bash

# Install pyyaml
pip install pyyaml==5.1

# Clone detectron2 repository
git clone 'https://github.com/facebookresearch/detectron2'

# Install detectron2 dependencies
cd detectron2
pip install $(python -c "import distutils.core; dist = distutils.core.run_setup('setup.py'); print(' '.join(dist.install_requires))")
cd ..

# Add detectron2 to Python path
export PYTHONPATH=$(pwd)/detectron2:$PYTHONPATH