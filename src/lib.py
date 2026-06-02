import librosa

print(librosa.__version__)

# src/test_audio.py

import pandas as pd
import soundfile as sf

df = pd.read_csv("data/processed_data/train.csv")

audio_path = df.iloc[0]["filepath"]

print(audio_path)

y, sr = sf.read(audio_path)

print("Loaded Successfully")
print("Sample Rate:", sr)
print("Shape:", y.shape)