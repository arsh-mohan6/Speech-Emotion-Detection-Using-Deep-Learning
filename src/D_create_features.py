import numpy as np
import pandas as pd

from tqdm import tqdm

from C_audio_utils import (
    load_audio,
    add_noise,
    pitch_shift,
    time_stretch,
    extract_features
)

emotion_map = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5
}

def process_audio(audio):

    mfcc, delta, delta2, mel = extract_features(
        audio
    )

    mfcc_features = np.vstack([
        mfcc,
        delta,
        delta2
    ])

    return (
        mfcc_features,
        mel
    )
    

def process_split(
    csv_path,
    augment=False
):

    df = pd.read_csv(
        csv_path
    )

    print(
        f"\nProcessing: {csv_path}"
    )

    X_mfcc = []
    X_mel = []
    y = []

    for idx, row in tqdm(
        df.iterrows(),
        total=len(df)
    ):

        audio = load_audio(
            row["filepath"]
        )

        if augment:

            audio_versions = [
                audio,
                add_noise(audio),
                pitch_shift(audio),
                time_stretch(audio)
            ]

        else:

            audio_versions = [
                audio
            ]

        for aug_audio in audio_versions:

            mfcc_features, mel_features = process_audio(
                aug_audio
            )

            X_mfcc.append(
                mfcc_features
            )

            X_mel.append(
                mel_features
            )

            y.append(
                emotion_map[
                    row["emotion"]
                ]
            )

    X_mfcc = np.array(
        X_mfcc,
        dtype=np.float32
    )

    X_mel = np.array(
        X_mel,
        dtype=np.float32
    )

    y = np.array(
        y,
        dtype=np.int32
    )

    return (
        X_mfcc,
        X_mel,
        y
    )
    
# ==========================================
# TRAIN
# ==========================================

X_train_mfcc, X_train_mel, y_train = process_split(
    "data/processed_data/train.csv",
    augment=True
)

# ==========================================
# VALIDATION
# ==========================================

X_val_mfcc, X_val_mel, y_val = process_split(
    "data/processed_data/val.csv",
    augment=False
)

# ==========================================
# TEST
# ==========================================

X_test_mfcc, X_test_mel, y_test = process_split(
    "data/processed_data/test.csv",
    augment=False
)
# ==========================================
# SAVE
# ==========================================


np.save(
    "data/processed_data/X_train_mfcc.npy",
    X_train_mfcc
)

np.save(
    "data/processed_data/X_train_mel.npy",
    X_train_mel
)

np.save(
    "data/processed_data/y_train.npy",
    y_train
)

np.save(
    "data/processed_data/X_val_mfcc.npy",
    X_val_mfcc
)

np.save(
    "data/processed_data/X_val_mel.npy",
    X_val_mel
)

np.save(
    "data/processed_data/y_val.npy",
    y_val
)

np.save(
    "data/processed_data/X_test_mfcc.npy",
    X_test_mfcc
)

np.save(
    "data/processed_data/X_test_mel.npy",
    X_test_mel
)

np.save(
    "data/processed_data/y_test.npy",
    y_test
)

print("\nTrain MFCC:", X_train_mfcc.shape)
print("Train Mel:", X_train_mel.shape)
print("Train Labels:", y_train.shape)

print("\nValidation MFCC:", X_val_mfcc.shape)
print("Validation Mel:", X_val_mel.shape)
print("Validation Labels:", y_val.shape)

print("\nTest MFCC:", X_test_mfcc.shape)
print("Test Mel:", X_test_mel.shape)
print("Test Labels:", y_test.shape)