from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2Model
)

import librosa
import torch

# ==========================================
# LOAD PROCESSOR & MODEL
# ==========================================

processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base"
)

model = Wav2Vec2Model.from_pretrained(
    "facebook/wav2vec2-base"
)

print("Loaded Successfully")

# ==========================================
# LOAD ONE AUDIO FILE
# ==========================================

audio_path = (
    "data/AudioWAV/1078_IWL_DIS_XX.wav"
)

audio, sr = librosa.load(
    audio_path,
    sr=16000
)

print("\nAudio Shape:")
print(audio.shape)

print("\nSample Rate:")
print(sr)

# ==========================================
# PROCESS AUDIO
# ==========================================

inputs = processor(
    audio,
    sampling_rate=16000,
    return_tensors="pt"
)

# ==========================================
# EXTRACT FEATURES
# ==========================================

with torch.no_grad():

    outputs = model(
        inputs.input_values
    )

print("\nLast Hidden State Shape:")
print(
    outputs.last_hidden_state.shape
)

pooled = outputs.last_hidden_state.mean(
    dim=1
)

print("\nPooled Shape:")
print(pooled.shape)