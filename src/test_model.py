# ==========================================
# IMPORTS
# ==========================================

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import to_categorical


# ==========================================
# CUSTOM ATTENTION LAYER
# ==========================================

class AttentionLayer(Layer):

    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)

    def build(
        self,
        input_shape
    ):

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

    def call(
        self,
        inputs
    ):

        score = tf.nn.tanh(
            tf.matmul(inputs, self.W)
            + self.b
        )

        attention_weights = tf.nn.softmax(
            score,
            axis=1
        )

        context_vector = (
            attention_weights
            * inputs
        )

        context_vector = tf.reduce_sum(
            context_vector,
            axis=1
        )

        return context_vector

    def get_config(self):

        config = super().get_config()

        return config


# ==========================================
# LOAD TEST DATA
# ==========================================

X_test = np.load(
    "data/processed_data/X_test.npy"
)

y_test = np.load(
    "data/processed_data/y_test.npy"
)

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

# ==========================================
# ADD CHANNEL DIMENSION
# ==========================================

X_test = X_test[..., np.newaxis]

print("New Shape:", X_test.shape)

# ==========================================
# LOAD MODEL
# ==========================================

model = load_model(
    "models/best_model.keras",
    custom_objects={
        "AttentionLayer": AttentionLayer
    }
)

print("Model Loaded Successfully")



y_test = to_categorical(
    y_test,
    6
)

# ==========================================
# TEST EVALUATION
# ==========================================

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print("\nTest Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

