SAMPLE_RATE = 22050
DURATION = 3
SAMPLES = SAMPLE_RATE * DURATION

N_MFCC = 40
N_MELS = 128
N_CHROMA = 12


import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

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

def add_noise(audio):
    noise = np.random.randn(len(audio))
    return audio + 0.005 * noise


def pitch_shift(audio):
    return librosa.effects.pitch_shift(
        audio,
        sr=SAMPLE_RATE,
        n_steps=2
    )


def time_stretch(audio):
    stretched = librosa.effects.time_stretch(
        audio,
        rate=0.9
    )

    if len(stretched) > SAMPLES:
        stretched = stretched[:SAMPLES]

    else:
        stretched = np.pad(
            stretched,
            (0, SAMPLES - len(stretched))
        )

    return stretched


def volume_change(audio):
    return audio * 1.5


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


emotion_map = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5
}

df = pd.read_csv("data/processed_data/train.csv")

audio_path = df.iloc[0]["filepath"]

print("Audio:", audio_path)

audio = load_audio(audio_path)

mfcc, mel, chroma = extract_features(audio)
 
print("MFCC Shape:", mfcc.shape)
print("Mel Shape:", mel.shape)
print("Chroma Shape:", chroma.shape)


features = np.vstack([
    mfcc,
    mel,
    chroma
])

print(features.shape)




# Testing 


train_df = pd.read_csv(
    "data/processed_data/train.csv"
)

X = []
y = []

for idx, row in tqdm(
    train_df.iterrows(),
    total=len(train_df)
):

    audio = load_audio(row["filepath"])

    audio_versions = [
        audio,
        add_noise(audio),
        pitch_shift(audio),
        time_stretch(audio),
        volume_change(audio)
    ]

    for aug_audio in audio_versions:

        mfcc, mel, chroma = extract_features(
            aug_audio
        )

        features = np.vstack([
            mfcc,
            mel,
            chroma
        ])

        X.append(features)

        y.append(
            emotion_map[row["emotion"]]
        )

X = np.array(X)
y = np.array(y)

print("X Shape:", X.shape)
print("y Shape:", y.shape)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

print("Final X Shape:", X.shape)
print("Final y Shape:", y.shape)

np.save(
    "data/processed_data/X_train.npy",
    X
)

np.save(
    "data/processed_data/y_train.npy",
    y
)

print("Train features saved.")

print(X.dtype)
print(y.dtype)