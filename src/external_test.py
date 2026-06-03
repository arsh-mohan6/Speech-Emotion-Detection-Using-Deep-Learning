import os
import numpy as np
import librosa
import tensorflow as tf
import joblib

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)
from audio_utils import (
    load_audio,
    extract_features
)
# ==========================================
# DATASET PATH
# ==========================================

DATASET_PATH = "data/Real_Data"

# ==========================================
# EMOTION MAPPING
# ==========================================

emotion_map = {
    "01": "neutral",
    "02": "happy",
    "04": "disgust"
}

# ==========================================
# COLLECT VALID FILES
# ==========================================

valid_files = []

for actor in os.listdir(DATASET_PATH):

    actor_path = os.path.join(
        DATASET_PATH,
        actor
    )

    if not os.path.isdir(actor_path):
        continue

    for file in os.listdir(actor_path):

        if not file.endswith(".wav"):
            continue

        parts = file.replace(
            ".wav",
            ""
        ).split("-")

        emotion_code = parts[1]

        if emotion_code in emotion_map:

            valid_files.append({
                "path": os.path.join(
                    actor_path,
                    file
                ),
                "emotion": emotion_map[
                    emotion_code
                ]
            })

# ==========================================
# SUMMARY
# ==========================================

print(
    "Total Valid Files:",
    len(valid_files)
)

emotion_count = {}

for item in valid_files:

    emotion = item["emotion"]

    emotion_count[emotion] = (
        emotion_count.get(
            emotion,
            0
        ) + 1
    )

print("\nEmotion Distribution:")

for emotion, count in emotion_count.items():

    print(
        f"{emotion}: {count}"
    )
    
    
class AttentionLayer(Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):

        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], 1),
            initializer="random_normal",
            trainable=True
        )

        self.b = self.add_weight(
            name="attention_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True
        )

        super().build(input_shape)

    def call(self, inputs):

        score = tf.nn.tanh(
            tf.matmul(inputs, self.W) + self.b
        )

        attention_weights = tf.nn.softmax(
            score,
            axis=1
        )

        context_vector = (
            attention_weights * inputs
        )

        return tf.reduce_sum(
            context_vector,
            axis=1
        )
        
model = load_model(
    "models/best_model.keras",
    custom_objects={
        "AttentionLayer": AttentionLayer
    }
)

scaler = joblib.load(
    "models/scaler.pkl"
)

print("Model Loaded")
print("Scaler Loaded")

# ==========================================
# TEST ONE FILE
# ==========================================

# ==========================================
# TEST 3 FILES
# ==========================================

test_files = [
    "data/Real_Data/Actor_01/01-01-01-01.wav",  # Neutral
    "data/Real_Data/Actor_01/01-02-01-01.wav",  # Happy
    "data/Real_Data/Actor_01/01-04-01-01.wav"   # Disgust
]

emotion_lookup = {
    "01": "neutral",
    "02": "happy",
    "04": "disgust"
}

model_emotions = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad"
]

for file_path in test_files:

    print("\n" + "=" * 50)

    filename = os.path.basename(
        file_path
    )

    emotion_code = filename.split("-")[1]

    actual_emotion = emotion_lookup[
        emotion_code
    ]

    print("File:", filename)
    print("Actual:", actual_emotion)

    # Load Audio
    audio = load_audio(
        file_path
    )

    # Extract Features
    mfcc, mel, chroma = extract_features(
        audio
    )

    features = np.vstack([
        mfcc,
        mel,
        chroma
    ])

    # Normalize
    features_2d = features.reshape(
        -1,
        features.shape[-1]
    )

    features_2d = scaler.transform(
        features_2d
    )

    features = features_2d.reshape(
        180,
        130
    )
    features_2d = scaler.transform(features_2d)

    print(features_2d.min())
    print(features_2d.max())
    print(features_2d.mean())

    # Model Input Shape
    features = np.expand_dims(
        features,
        axis=0
    )

    features = np.expand_dims(
        features,
        axis=-1
    )

    # Prediction
    prediction = model.predict(
        features,
        verbose=0
    )

    predicted_class = np.argmax(
        prediction
    )

    predicted_emotion = model_emotions[
        predicted_class
    ]

    confidence = np.max(
        prediction
    )

    print(
        "Predicted:",
        predicted_emotion
    )

    print(
        "Confidence:",
        f"{confidence:.4f}"
    )
print("\nRaw Feature Stats")
print("Min :", features.min())
print("Max :", features.max())
print("Mean:", features.mean())