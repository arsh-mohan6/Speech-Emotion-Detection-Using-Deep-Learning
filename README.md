# Speech Emotion Detection Using Deep Learning

This project focuses on detecting human emotions from speech audio using deep learning techniques. The RAVDESS dataset is used for training and evaluation.


Speech-Emotion-Detection/

├── data/
│   ├── AudioWAV/                           # CREMA-D Dataset
│   ├── Radvess/                            # RAVDESS Dataset
│   ├── TESS Toronto emotional speech set data/   # TESS Dataset
│   │
│   └── processed_data/
│       ├── combined_dataset.csv            # Combined metadata of all datasets
│       ├── train.csv                       # Training split
│       ├── val.csv                         # Validation split
│       └── test.csv                        # Test split
│
├── src/
│   ├── combine_datasets.py                 # Merge RAVDESS, CREMA-D, TESS
│   ├── split_dataset.py                    # Train/Validation/Test split
│   ├── feature_extraction.py               # Audio preprocessing & feature extraction
│   ├── train_model.py                      # CNN-BiLSTM-Attention model training
│   └── predict.py                          # Emotion prediction from audio
│
├── models/
│   └── best_model.h5                       # Trained model
│
├── requirements.txt
├── README.md
└── .gitignore

Pipeline:

RAVDESS + CREMA-D + TESS
│
▼
Dataset Combination
│
▼
Train / Validation / Test Split
│
▼
Audio Preprocessing
(Resampling, Padding, Trimming)
│
▼
Data Augmentation (Train Only)
• Noise Injection
• Pitch Shift
• Time Stretch
• Volume Change
│
▼
Feature Extraction
• MFCC (40)
• Mel Spectrogram (128)
• Chroma Features (12)
│
▼
Feature Fusion
(180 × 130 Feature Matrix)
│
▼
CNN Layers
(Spatial Feature Learning)
│
▼
Bi-LSTM Layers
(Temporal Dependency Learning)
│
▼
Attention Mechanism
(Focus on Important Emotional Segments)
│
▼
Dense + Softmax
│
▼
Emotion Classification

Target Emotions:
• Angry
• Disgust
• Fear
• Happy
• Neutral
• Sad

Datasets Used:
• RAVDESS (Speech Only)
• CREMA-D
• TESS

Total Samples Before Augmentation:
10,898

Training Samples:
8,718

Validation Samples:
1,090

Test Samples:
1,090
