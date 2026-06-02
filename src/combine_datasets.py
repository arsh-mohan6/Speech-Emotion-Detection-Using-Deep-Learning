import os
import pandas as pd

data = []

COMMON_EMOTIONS = {
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad"
}

ravdess_emotions = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise"
}

for root, dirs, files in os.walk("data/Radvess"):

    for file in files:

        if file.endswith(".wav"):

            emotion_code = file.split("-")[2]
            emotion = ravdess_emotions[emotion_code]

            if emotion in COMMON_EMOTIONS:

                data.append({
                    "filepath": os.path.join(root, file),
                    "emotion": emotion,
                    "dataset": "RAVDESS"
                })


crema_emotions = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad"
}

for file in os.listdir("data/AudioWAV"):

    if file.endswith(".wav"):

        code = file.split("_")[2]
        emotion = crema_emotions[code]

        data.append({
            "filepath": os.path.join("data/AudioWAV", file),
            "emotion": emotion,
            "dataset": "CREMA-D"
        })
        
        
tess_root = "data/TESS Toronto emotional speech set data"

for folder in os.listdir(tess_root):

    folder_path = os.path.join(tess_root, folder)

    if not os.path.isdir(folder_path):
        continue

    emotion = folder.split("_")[-1].lower()

    if emotion == "pleasant":
        continue

    if emotion == "pleasant_surprise":
        continue

    if emotion not in COMMON_EMOTIONS:
        continue

    for file in os.listdir(folder_path):

        if file.endswith(".wav"):

            data.append({
                "filepath": os.path.join(folder_path, file),
                "emotion": emotion,
                "dataset": "TESS"
            })
df = pd.DataFrame(data)

print("\nDataset Counts:")
print(df["dataset"].value_counts())

print("\nEmotion Counts:")
print(df["emotion"].value_counts())

print("\nTotal Samples:")
print(len(df))

df.to_csv("combined_dataset.csv", index=False)

print("\ncombined_dataset.csv saved successfully.")