import os
import lib
import soundfile as sf
import numpy as np
import pandas as pd

# Load train.csv
train_df = pd.read_csv(
    "data/processed_data/train.csv"
)

# Take first sample only
audio_path = train_df.iloc[0]["filepath"]
emotion = train_df.iloc[0]["emotion"]

print("Audio:", audio_path)
print("Emotion:", emotion)

# Output folder
save_dir = os.path.join(
    "data",
    "processed_data",
    "augmented_audio",
    emotion
)

os.makedirs(save_dir, exist_ok=True)

# Load audio
y, sr = lib.load(audio_path, sr=None)

# -------------------
# Original
# -------------------
sf.write(
    os.path.join(save_dir, "original.wav"),
    y,
    sr
)

# -------------------
# Noise
# -------------------
noise = y + 0.005 * np.random.randn(len(y))

sf.write(
    os.path.join(save_dir, "noise.wav"),
    noise,
    sr
)

# -------------------
# Pitch Shift
# -------------------
pitch = lib.effects.pitch_shift(
    y,
    sr=sr,
    n_steps=2
)

sf.write(
    os.path.join(save_dir, "pitch.wav"),
    pitch,
    sr
)

# -------------------
# Time Stretch
# -------------------
stretch = lib.effects.time_stretch(
    y,
    rate=0.9
)

sf.write(
    os.path.join(save_dir, "stretch.wav"),
    stretch,
    sr
)

# -------------------
# Volume Change
# -------------------
volume = y * 1.5

sf.write(
    os.path.join(save_dir, "volume.wav"),
    volume,
    sr
)

print("Augmentation completed.")