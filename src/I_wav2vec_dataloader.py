import torch

from torch.utils.data import (
    DataLoader
)

from transformers import (
    Wav2Vec2Processor
)

from H_wav2vec_dataset import (
    SERDataset
)

# ==========================================
# COLLATE FUNCTION
# ==========================================

def collate_fn(batch):

    input_values = [
        item["input_values"]
        for item in batch
    ]

    labels = torch.tensor(
        [
            item["label"]
            for item in batch
        ]
    )

    padded_inputs = torch.nn.utils.rnn.pad_sequence(
        input_values,
        batch_first=True,
        padding_value=0
    )

    attention_masks = [
        item["attention_mask"]
        for item in batch
    ]

    padded_masks = torch.nn.utils.rnn.pad_sequence(
        attention_masks,
        batch_first=True,
        padding_value=0
    )

    return {
        "input_values":
            padded_inputs,

        "attention_mask":
            padded_masks,

        "labels":
            labels
    }

# ==========================================
# TEST
# ==========================================

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

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn
    )

    batch = next(
        iter(dataloader)
    )

    print(
        batch["input_values"].shape
    )

    print(
        batch["labels"].shape
    )
    
    from transformers import Wav2Vec2Model

    model = Wav2Vec2Model.from_pretrained(
        "facebook/wav2vec2-base"
    )

    batch = next(iter(dataloader))

    with torch.no_grad():

        outputs = model(
            batch["input_values"],
            attention_mask=batch["attention_mask"]
        )

    print(
        outputs.last_hidden_state.shape
    )