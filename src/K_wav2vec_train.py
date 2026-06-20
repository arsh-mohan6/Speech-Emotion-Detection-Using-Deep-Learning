import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from transformers import Wav2Vec2Processor

from H_wav2vec_dataset import SERDataset
from I_wav2vec_dataloader import collate_fn
from J_wav2vec_model import Wav2Vec2SER

from L_wav2vec_config import *
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
# ==========================================
# DEVICE
# ==========================================

device = torch.device("cpu")

print("Device:", device)

# ==========================================
# PROCESSOR
# ==========================================

processor = Wav2Vec2Processor.from_pretrained(
    "facebook/wav2vec2-base"
)

# ==========================================
# DATASET
# ==========================================

train_dataset = SERDataset(
    "data/processed_data/train.csv",
    processor
)

val_dataset = SERDataset(
    "data/processed_data/val.csv",
    processor
)

# ==========================================
# DATALOADER
# ==========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)

print("Data Loaded")

# ==========================================
# CLASS WEIGHTS
# ==========================================

train_labels = [
    train_dataset.df.iloc[i]["emotion"]
    for i in range(len(train_dataset))
]

emotion_map = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5
}

train_labels = [
    emotion_map[label]
    for label in train_labels
]

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float
).to(device)

print("\nClass Weights:")
print(class_weights)

# ==========================================
# MODEL
# ==========================================

model = Wav2Vec2SER().to(device)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = torch.optim.AdamW(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=LEARNING_RATE
)

# ==========================================
# TRAINING
# ==========================================

best_val_acc = 0

for epoch in range(EPOCHS):

    model.train()

    train_correct = 0
    train_total = 0
    train_loss = 0

    for batch in train_loader:

        inputs = batch["input_values"].to(device)
        
        attention_mask = batch[
            "attention_mask"
        ].to(device)


        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(
            inputs,
        attention_mask
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        train_loss += loss.item()

        preds = outputs.argmax(dim=1)

        train_correct += (
            preds == labels
        ).sum().item()

        train_total += labels.size(0)

    train_acc = (
        train_correct / train_total
    )

    # ======================================
    # VALIDATION
    # ======================================

    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for batch in val_loader:

            inputs = batch["input_values"].to(device)

            attention_mask = batch[
                "attention_mask"
            ].to(device)

            labels = batch["labels"].to(device)

            outputs = model(
                inputs,
                attention_mask
            )

            preds = outputs.argmax(
                dim=1
            )

            val_correct += (
                preds == labels
            ).sum().item()

            val_total += labels.size(0)

    val_acc = (
        val_correct / val_total
    )

    print(
        f"Epoch {epoch+1}/{EPOCHS}"
    )

    print(
        f"Train Acc: {train_acc:.4f}"
    )

    print(
        f"Val Acc: {val_acc:.4f}"
    )

    # ======================================
    # SAVE BEST MODEL
    # ======================================

    if val_acc > best_val_acc:

        best_val_acc = val_acc

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print(
            "Best Model Saved"
        )
        print(
            f"Best Model Saved | Val Acc: {val_acc:.4f}"
        )
        

print("\nTraining Finished")