import numpy as np
import tensorflow as tf

X_test_mfcc = np.load(
    "data/processed_data/X_test_mfcc.npy"
)

X_test_mel = np.load(
    "data/processed_data/X_test_mel.npy"
)

y_test = np.load(
    "data/processed_data/y_test.npy"
)

print("MFCC:", X_test_mfcc.shape)
print("Mel :", X_test_mel.shape)
print("y   :", y_test.shape)

print("\nLabels Distribution")

unique, counts = np.unique(
    y_test,
    return_counts=True
)

print(
    dict(
        zip(unique, counts)
    )
)

import joblib

scaler_mfcc = joblib.load(
    "models/scaler_mfcc.pkl"
)

scaler_mel = joblib.load(
    "models/scaler_mel.pkl"
)

print("Scalers Loaded")

# ==========================================
# MFCC SCALING
# ==========================================

X_test_mfcc_2d = X_test_mfcc.reshape(
    -1,
    X_test_mfcc.shape[-1]
)

X_test_mfcc_2d = scaler_mfcc.transform(
    X_test_mfcc_2d
)

X_test_mfcc = X_test_mfcc_2d.reshape(
    X_test_mfcc.shape
)

# ==========================================
# MEL LOG TRANSFORM
# ==========================================

X_test_mel = np.log1p(
    X_test_mel
)

# ==========================================
# MEL SCALING
# ==========================================

X_test_mel_2d = X_test_mel.reshape(
    -1,
    X_test_mel.shape[-1]
)

X_test_mel_2d = scaler_mel.transform(
    X_test_mel_2d
)

X_test_mel = X_test_mel_2d.reshape(
    X_test_mel.shape
)

# ==========================================
# CHANNEL DIMENSION
# ==========================================

X_test_mel = X_test_mel[
    ...,
    np.newaxis
]

print("\nMFCC")
print(X_test_mfcc.shape)
print(np.mean(X_test_mfcc))
print(np.std(X_test_mfcc))

print("\nMel")
print(X_test_mel.shape)
print(np.mean(X_test_mel))
print(np.std(X_test_mel))




class AttentionLayer(tf.keras.layers.Layer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):

        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
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

        context_vector = tf.reduce_sum(
            context_vector,
            axis=1
        )

        return context_vector
    
    
    
    
from tensorflow.keras.models import load_model

model = load_model(
    "models/best_model.keras",
    custom_objects={
        "AttentionLayer": AttentionLayer
    }
)

print("Model Loaded")

print(model.inputs)
print(model.outputs)


y_test_cat = tf.keras.utils.to_categorical(
    y_test,
    6
)

results = model.evaluate(
    [
        X_test_mfcc,
        X_test_mel
    ],
    y_test_cat,
    verbose=1
)

print("\nResults:")
print(results)
