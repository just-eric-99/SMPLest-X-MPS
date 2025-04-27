#!/usr/bin/env bash
conda create -n smplestx-v2 python=3.8 -y
conda activate smplestx-v2
pip install pytorch torchvision torchaudio -c pytorch -y
pip install -r requirements.txt

# update version for pyopengl
# Note: please ignore the incompatible error message if 3.1.4 can be installed
pip install pyopengl==3.1.4