import numpy as np
import librosa

SAMPLE_RATE = 22050
DURATION = 3
SAMPLES = SAMPLE_RATE * DURATION

N_MFCC = 40
N_MELS = 128


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

    noise = np.random.randn(
        len(audio)
    )

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


def extract_features(audio):

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC
    )

    delta_mfcc = librosa.feature.delta(
        mfcc
    )

    delta2_mfcc = librosa.feature.delta(
        mfcc,
        order=2
    )

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=N_MELS
    )

    return (
        mfcc,
        delta_mfcc,
        delta2_mfcc,
        mel
    )