# Speech Emotion Detection Using Deep Learning

This project focuses on detecting human emotions from speech audio using deep learning techniques. The RAVDESS dataset is used for training and evaluation.

Successfully generated 43,590 augmented training samples
from 8,718 original training recordings using five
augmentation strategies and feature fusion of MFCC,
Mel Spectrogram and Chroma features.


Feature Extraction
↓
StandardScaler (fit on train only)
↓
CNN + BatchNorm
↓
CNN + BatchNorm
↓
BiLSTM(128)
↓
Attention
↓
Dense(64)
↓
Dropout(0.3)
↓
Softmax(6)