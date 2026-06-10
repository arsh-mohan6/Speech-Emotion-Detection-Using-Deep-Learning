from C_audio_utils import *

audio = load_audio(
    r"data\Radvess\Actor_01\03-01-01-01-01-01-01.wav"
)
mfcc, delta, delta2, mel = extract_features(
    audio
)

print(mfcc.shape)
print(delta.shape)
print(delta2.shape)
print(mel.shape)