# ==========================================
# IMPORT LIBRARIES
# ==========================================

import numpy as np

from sklearn.preprocessing import StandardScaler

import tensorflow as tf

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Model

from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    MaxPooling2D,
    Reshape,
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
    Layer
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)


# ==========================================
# LOAD TRAIN / VAL / TEST DATA
# ==========================================

X_train = np.load(
    "data/processed_data/X_train.npy"
)

y_train = np.load(
    "data/processed_data/y_train.npy"
)

X_val = np.load(
    "data/processed_data/X_val.npy"
)

y_val = np.load(
    "data/processed_data/y_val.npy"
)

X_test = np.load(
    "data/processed_data/X_test.npy"
)

y_test = np.load(
    "data/processed_data/y_test.npy"
)

print("Train:", X_train.shape, y_train.shape)
print("Val:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)

# ==========================================
# FEATURE NORMALIZATION
# FIT ONLY ON TRAIN DATA
# ==========================================

scaler = StandardScaler()

X_train_2d = X_train.reshape(
    -1,
    X_train.shape[-1]
)

X_val_2d = X_val.reshape(
    -1,
    X_val.shape[-1]
)

X_test_2d = X_test.reshape(
    -1,
    X_test.shape[-1]
)

scaler.fit(X_train_2d)

X_train_2d = scaler.transform(X_train_2d)
X_val_2d = scaler.transform(X_val_2d)
X_test_2d = scaler.transform(X_test_2d)

X_train = X_train_2d.reshape(
    X_train.shape
)

X_val = X_val_2d.reshape(
    X_val.shape
)

X_test = X_test_2d.reshape(
    X_test.shape
)

print("Normalization Complete")

# ==========================================
# ADD CHANNEL DIMENSION
# ==========================================

X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]
X_test = X_test[..., np.newaxis]

print("Train Shape:", X_train.shape)
print("Val Shape:", X_val.shape)
print("Test Shape:", X_test.shape)

# ==========================================
# ONE HOT ENCODE LABELS
# ==========================================

NUM_CLASSES = 6

y_train = to_categorical(
    y_train,
    NUM_CLASSES
)

y_val = to_categorical(
    y_val,
    NUM_CLASSES
)

y_test = to_categorical(
    y_test,
    NUM_CLASSES
)

print("y_train:", y_train.shape)
print("y_val:", y_val.shape)
print("y_test:", y_test.shape)

# ==========================================
# CUSTOM ATTENTION LAYER
# ==========================================

class AttentionLayer(Layer):

    def __init__(self):
        super(
            AttentionLayer,
            self
        ).__init__()

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
            tf.matmul(
                inputs,
                self.W
            ) + self.b
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

# ==========================================
# MODEL INPUT
# ==========================================

inputs = Input(
    shape=(180, 130, 1)
)

# ==========================================
# CNN BLOCK 1
# ==========================================

x = Conv2D(
    filters=32,
    kernel_size=(3, 3),
    activation="relu",
    padding="same"
)(inputs)

x = BatchNormalization()(x)

x = MaxPooling2D(
    pool_size=(2, 2)
)(x)

# ==========================================
# CNN BLOCK 2
# ==========================================

x = Conv2D(
    filters=64,
    kernel_size=(3, 3),
    activation="relu",
    padding="same"
)(x)

x = BatchNormalization()(x)

x = MaxPooling2D(
    pool_size=(2, 2)
)(x)

# ==========================================
# RESHAPE FOR BiLSTM
# ==========================================

x = Reshape(
    target_shape=(45, 32 * 64)
)(x)

# ==========================================
# BiLSTM
# ==========================================

x = Bidirectional(
    LSTM(
        64,
        return_sequences=True
    )
)(x)


# ==========================================
# ATTENTION
# ==========================================

x = AttentionLayer()(x)

# ==========================================
# CLASSIFICATION HEAD
# ==========================================

x = Dense(
    64,
    activation="relu"
)(x)

x = Dropout(
    0.3
)(x)


# ==========================================
# OUTPUT LAYER
# ==========================================

outputs = Dense(
    6,
    activation="softmax"
)(x)

# ==========================================
# BUILD MODEL
# ==========================================

model = Model(
    inputs=inputs,
    outputs=outputs
)

model.summary()

# ==========================================
# COMPILE MODEL
# ==========================================

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ==========================================
# CALLBACKS
# ==========================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    verbose=1
)

checkpoint = ModelCheckpoint(
    "models/best_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)
print("\nStarting Training...\n")
# ==========================================
# TRAIN MODEL
# ==========================================

history = model.fit(
    X_train,
    y_train,
    validation_data=(
        X_val,
        y_val
    ),
    epochs=15,
    batch_size=64,
    callbacks=[
        early_stopping,
        reduce_lr,
        checkpoint
    ]
)