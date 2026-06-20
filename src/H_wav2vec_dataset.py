import pandas as pd
import librosa
import torch

from torch.utils.data import Dataset

from transformers import (
    Wav2Vec2Processor
)

# ==========================================
# LABEL MAP
# ==========================================

emotion_map = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5
}

# ==========================================
# DATASET
# ==========================================

class SERDataset(Dataset):

    def __init__(
        self,
        csv_file,
        processor
    ):

        self.df = pd.read_csv(
            csv_file
        )

        self.processor = processor

    def __len__(self):

        return len(
            self.df
        )

    def __getitem__(
        self,
        idx
    ):

        row = self.df.iloc[idx]

        filepath = row["filepath"]

        emotion = row["emotion"]

        label = emotion_map[
            emotion
        ]

        # ==================================
        # LOAD AUDIO
        # ==================================

        audio, sr = librosa.load(
            filepath,
            sr=16000
        )

        # ==================================
        # PROCESS AUDIO
        # ==================================

        inputs = self.processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True
        )

        return {
            "input_values":
                inputs.input_values.squeeze(0),

            "attention_mask":
                inputs.attention_mask.squeeze(0),

            "label":
                torch.tensor(
                    label,
                    dtype=torch.long
                )
        }
        
if __name__ == "__main__":

    processor = (
        Wav2Vec2Processor
        .from_pretrained(
            "facebook/wav2vec2-base"
        )
    )

    dataset = SERDataset(
        "data/processed_data/train.csv",
        processor
    )

    sample = dataset[0]

    print(
        sample["input_values"].shape
    )

    print(
        sample["label"]
    )