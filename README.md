# Speech Emotion Detection Using Deep Learning

<<<<<<< HEAD
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
=======
This project focuses on detecting human emotions from speech audio using Deep Learning techniques. The model is trained using a combination of three popular emotional speech datasets: **RAVDESS**, **CREMA-D**, and **TESS**.

The objective is to classify speech into six emotional categories using a **CNN + Bi-LSTM + Attention** based architecture.

---

# Project Structure

```text
Speech-Emotion-Detection/

├── data/
│   ├── AudioWAV/                                 # CREMA-D Dataset
│   ├── Radvess/                                  # RAVDESS Dataset
│   ├── TESS Toronto emotional speech set data/  # TESS Dataset
│   │
│   └── processed_data/
│       ├── combined_dataset.csv
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
│
├── src/
│   ├── combine_datasets.py
│   ├── split_dataset.py
│   ├── feature_extraction.py
│   ├── train_model.py
│   └── predict.py
│
├── models/
│   └── best_model.h5
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Dataset Information

## Datasets Used

* RAVDESS (Speech Only)
* CREMA-D
* TESS

## Target Emotions

* Angry
* Disgust
* Fear
* Happy
* Neutral
* Sad

---

# Data Statistics

## Dataset Integration Results

| Dataset   |    Samples |
| --------- | ---------: |
| RAVDESS   |      1,056 |
| CREMA-D   |      7,442 |
| TESS      |      2,400 |
| **Total** | **10,898** |

---

## Emotion Distribution

| Emotion | Samples |
| ------- | ------: |
| Angry   |   1,863 |
| Disgust |   1,863 |
| Fear    |   1,863 |
| Happy   |   1,863 |
| Sad     |   1,863 |
| Neutral |   1,583 |

---

## Dataset Split

| Split      | Samples |
| ---------- | ------: |
| Training   |   8,718 |
| Validation |   1,090 |
| Test       |   1,090 |

---

# Proposed Pipeline

```text
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
```

---

# Feature Extraction Configuration

| Parameter       | Value     |
| --------------- | --------- |
| Sample Rate     | 22,050 Hz |
| Audio Duration  | 3 Seconds |
| MFCC Features   | 40        |
| Mel Bands       | 128       |
| Chroma Features | 12        |

---

# Feature Extraction Results

Successfully tested on sample audio files.

| Feature           | Shape      |
| ----------------- | ---------- |
| MFCC              | (40, 130)  |
| Mel Spectrogram   | (128, 130) |
| Chroma            | (12, 130)  |
| Combined Features | (180, 130) |

---

# Data Augmentation Strategy

Applied only on the Training Dataset.

1. Original Audio
2. Noise Injection
3. Pitch Shift
4. Time Stretch
5. Volume Change

Expected Training Samples After Augmentation:

```text
8,718 × 5 = 43,590 Samples
```

---

# Feature Extraction Validation

Tested on 10 audio samples.

Output:

```text
X Shape : (50, 180, 130)
y Shape : (50,)
```

This validates:

* Audio Loading
* Audio Preprocessing
* Data Augmentation
* MFCC Extraction
* Mel Spectrogram Extraction
* Chroma Extraction
* Feature Fusion
* Label Encoding

---

# Current Progress

| Task                        | Status |
| --------------------------- | ------ |
| Dataset Collection          |   D    |
| Dataset Combination         |   D    |
| Dataset Splitting           |   D    |
| Audio Preprocessing         |   D    |
| Data Augmentation Pipeline  |   D    |
| Feature Extraction Pipeline |   D    |
| CNN Model                   |   R    |
| Bi-LSTM                     |   R    |
| Attention Layer             |   R    |
| Model Training              |   R    |
| Model Evaluation            |   R    |
| Real-Time Prediction        |   R    |

---

# Future Work

* Train CNN model on extracted features

* Add Attention mechanism
* Evaluate using Accuracy, Precision, Recall and F1-Score
* Deploy real-time speech emotion prediction system
* Compare performance across different feature combinations

---
>>>>>>> 1dc2f6691bdd770caaa230a42ed4f9a18d96f03e

                    INPUTS
                       │
        ┌──────────────┴──────────────┐
        │                             │
        ▼                             ▼

   MFCC Branch                  Mel Branch
 (120 × 130)                 (128 × 130 × 1)

        │                             │
        ▼                             ▼

  BiLSTM(64)                Conv2D(32)+BN
        │                             │
     Dropout                    MaxPool
        │                             │
  BiLSTM(32)                Conv2D(64)+BN
        │                             │
        ▼                       MaxPool
 MultiHeadAttn                     │
 (2 Heads)                         ▼
        │                    Conv2D(128)+BN
 Residual Add                      │
        │                           ▼
 LayerNorm                  GlobalAvgPool2D
        │                           │
 GlobalAvgPool1D                    │
        │                           │
        └──────────┬────────────────┘
                   │
                   ▼

              Concatenate
                   │
              Dropout
                   │
         Dense(128) + L2
                   │
              BN + ReLU
                   │
              Dropout
                   │
          Dense(64) + L2
                   │
              BN + ReLU
                   │
                   ▼

       Dense(6) + Softmax + L2
                   │
                   ▼

        Emotion Classification
 (Angry, Disgust, Fear, Happy,
        Neutral, Sad)
