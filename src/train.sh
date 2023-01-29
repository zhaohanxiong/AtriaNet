#!/bin/sh

# preprocess
python preprocess_ROI.py
python preprocess_Seg.py

# train model
python train_ROI.py
python train_Seg.py
