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