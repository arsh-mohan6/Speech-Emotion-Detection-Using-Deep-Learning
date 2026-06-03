import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

# =========================
# CONFIG
# =========================

SAMPLE_RATE = 22050
DURATION = 3
SAMPLES = SAMPLE_RATE * DURATION

N_MFCC = 40
N_MELS = 128
N_CHROMA = 12

# =========================
# AUDIO LOADING
# =========================

def load_audio(file_path):

    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE
    )

    if len(audio) > SAMPLES:
        audio = audio[:SAMPLES]

    else:
        audio = np.pad(
            audio,
            (0, SAMPLES - len(audio))
        )

    return audio


# =========================
# FEATURE EXTRACTION
# =========================

def extract_features(audio):

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC
    )

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=N_MELS
    )

    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=SAMPLE_RATE
    )

    return mfcc, mel, chroma


# =========================
# LABEL ENCODING
# =========================

emotion_map = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5
}


# =========================
# VALIDATION FEATURES
# =========================

print("\nGenerating Validation Features...")

val_df = pd.read_csv(
    "data/processed_data/val.csv"
)

X_val = []
y_val = []

for idx, row in tqdm(
    val_df.iterrows(),
    total=len(val_df)
):

    audio = load_audio(
        row["filepath"]
    )

    mfcc, mel, chroma = extract_features(
        audio
    )

    features = np.vstack([
        mfcc,
        mel,
        chroma
    ])

    X_val.append(features)

    y_val.append(
        emotion_map[row["emotion"]]
    )

X_val = np.array(
    X_val,
    dtype=np.float32
)

y_val = np.array(
    y_val,
    dtype=np.int32
)

print("X_val Shape:", X_val.shape)
print("y_val Shape:", y_val.shape)

np.save(
    "data/processed_data/X_val.npy",
    X_val
)

np.save(
    "data/processed_data/y_val.npy",
    y_val
)

print("Validation features saved.")


# =========================
# TEST FEATURES
# =========================

print("\nGenerating Test Features...")

test_df = pd.read_csv(
    "data/processed_data/test.csv"
)

X_test = []
y_test = []

for idx, row in tqdm(
    test_df.iterrows(),
    total=len(test_df)
):

    audio = load_audio(
        row["filepath"]
    )

    mfcc, mel, chroma = extract_features(
        audio
    )

    features = np.vstack([
        mfcc,
        mel,
        chroma
    ])

    X_test.append(features)

    y_test.append(
        emotion_map[row["emotion"]]
    )

X_test = np.array(
    X_test,
    dtype=np.float32
)

y_test = np.array(
    y_test,
    dtype=np.int32
)

print("X_test Shape:", X_test.shape)
print("y_test Shape:", y_test.shape)

np.save(
    "data/processed_data/X_test.npy",
    X_test
)

np.save(
    "data/processed_data/y_test.npy",
    y_test
)

print("Test features saved.")

print("\nAll validation and test features generated successfully.")