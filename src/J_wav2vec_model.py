import torch
import torch.nn as nn

from transformers import (
    Wav2Vec2Model
)

# ==========================================
# MODEL
# ==========================================

class Wav2Vec2SER(nn.Module):

    def __init__(self):

        super().__init__()

        self.wav2vec = (
            Wav2Vec2Model
            .from_pretrained(
                "facebook/wav2vec2-base"
            )
        )

        # Freeze Backbone

        for param in self.wav2vec.parameters():

            param.requires_grad = False

        self.classifier = nn.Sequential(

            nn.Linear(
                768,
                256
            ),

            nn.ReLU(),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                128,
                6
            )
        )

    def forward(
        self,
        input_values,
        attention_mask
    ):

        outputs = self.wav2vec(
            input_values,
            attention_mask=attention_mask
        )
        

        hidden_states = (
            outputs
            .last_hidden_state
        )

        pooled = hidden_states.mean(
            dim=1
        )

        logits = self.classifier(
            pooled
        )

        return logits
    

if __name__ == "__main__":

    model = Wav2Vec2SER()

    x = torch.randn(
        4,
        40574
    )

    mask = torch.ones_like(
        x,
        dtype=torch.long
    )

    outputs = model(
        x,
        mask
    )

    print(
        outputs.shape
    )