from pathlib import Path
import pandas as pd

# Dataset path
DATASET_PATH = Path("data/raw/ravdess")

print("Dataset path:", DATASET_PATH)
print("Path exists:", DATASET_PATH.exists())

# Scan all files
all_files = list(DATASET_PATH.rglob("*"))
files = [file for file in all_files if file.is_file()]

print("Total files found:", len(files))

# Check extensions
extensions = {}

for file in files:
    ext = file.suffix.lower()
    extensions[ext] = extensions.get(ext, 0) + 1

print("File extensions count:")
print(extensions)


# ==============================
# RAVDESS Label Mappings
# ==============================

modality_map = {
    "01": "audio_video",
    "02": "video_only",
    "03": "audio_only"
}

vocal_channel_map = {
    "01": "speech",
    "02": "song"
}

emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

intensity_map = {
    "01": "normal",
    "02": "strong"
}

statement_map = {
    "01": "Kids are talking by the door",
    "02": "Dogs are sitting by the door"
}

# ==============================
# Create Metadata
# ==============================

metadata = []

for file in files:
    if file.suffix.lower() not in [".wav", ".mp4"]:
        continue

    filename = file.stem
    parts = filename.split("-")

    if len(parts) != 7:
        continue

    modality, vocal_channel, emotion, intensity, statement, repetition, actor = parts

    actor_num = int(actor)
    gender = "male" if actor_num % 2 != 0 else "female"

    metadata.append({
        "file_path": str(file),
        "file_name": file.name,
        "extension": file.suffix.lower(),

        "modality_code": modality,
        "modality": modality_map.get(modality),

        "vocal_channel_code": vocal_channel,
        "vocal_channel": vocal_channel_map.get(vocal_channel),

        "emotion_code": emotion,
        "emotion": emotion_map.get(emotion),

        "intensity_code": intensity,
        "intensity": intensity_map.get(intensity),

        "statement_code": statement,
        "statement": statement_map.get(statement),

        "repetition": repetition,
        "actor": actor_num,
        "gender": gender
    })

metadata_df = pd.DataFrame(metadata)

print("\nMetadata shape:", metadata_df.shape)

print("\nMetadata sample:")
print(metadata_df.head())

# ==============================
# Basic Counts
# ==============================

print("\nModality count:")
print(metadata_df["modality"].value_counts())

print("\nVocal channel count:")
print(metadata_df["vocal_channel"].value_counts())

print("\nEmotion count:")
print(metadata_df["emotion"].value_counts())

print("\nGender count:")
print(metadata_df["gender"].value_counts())

# ==============================
# Filter Data for Our Project
# ==============================

speech_df = metadata_df[
    (metadata_df["modality"].isin(["audio_video", "audio_only"])) &
    (metadata_df["vocal_channel"] == "speech")
].copy()

print("\nFiltered speech dataset shape:", speech_df.shape)

print("\nFiltered modality count:")
print(speech_df["modality"].value_counts())

print("\nFiltered emotion count:")
print(speech_df["emotion"].value_counts())

# ==============================
# Save Metadata
# ==============================

METADATA_PATH = Path("data/metadata")
METADATA_PATH.mkdir(parents=True, exist_ok=True)

metadata_df.to_csv(METADATA_PATH / "ravdess_all_metadata.csv", index=False)
speech_df.to_csv(METADATA_PATH / "ravdess_speech_metadata.csv", index=False)

print("\nMetadata saved successfully!")
print("Saved files:")
print(METADATA_PATH / "ravdess_all_metadata.csv")
print(METADATA_PATH / "ravdess_speech_metadata.csv")