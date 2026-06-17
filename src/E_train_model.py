# ==========================================
# IMPORT LIBRARIES
# ==========================================

import numpy as np
import tensorflow as tf
import joblib

from sklearn.preprocessing import StandardScaler

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Model

from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    MaxPooling2D,
    GlobalAveragePooling2D,
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
    Concatenate,
    ReLU,
    Layer
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)
from tensorflow.keras.regularizers import l2

from a_config import *
# ==========================================
# LOAD DATA
# ==========================================

X_train_mfcc = np.load(
    "data/processed_data/X_train_mfcc.npy"
)

X_train_mel = np.load(
    "data/processed_data/X_train_mel.npy"
)

y_train = np.load(
    "data/processed_data/y_train.npy"
)

X_val_mfcc = np.load(
    "data/processed_data/X_val_mfcc.npy"
)

X_val_mel = np.load(
    "data/processed_data/X_val_mel.npy"
)

y_val = np.load(
    "data/processed_data/y_val.npy"
)

X_test_mfcc = np.load(
    "data/processed_data/X_test_mfcc.npy"
)

X_test_mel = np.load(
    "data/processed_data/X_test_mel.npy"
)

y_test = np.load(
    "data/processed_data/y_test.npy"
)

# print("\nMFCC Train:", X_train_mfcc.shape)
# print("Mel Train :", X_train_mel.shape)
# print("y Train   :", y_train.shape)

# print("\nMFCC Val:", X_val_mfcc.shape)
# print("Mel Val :", X_val_mel.shape)
# print("y Val   :", y_val.shape)

# print("\nMFCC Test:", X_test_mfcc.shape)
# print("Mel Test :", X_test_mel.shape)
# print("y Test   :", y_test.shape)

# ==========================================
# MFCC SCALING
# ==========================================

scaler_mfcc = StandardScaler()

X_train_mfcc_2d = X_train_mfcc.reshape(
    -1,
    X_train_mfcc.shape[-1]
)

X_val_mfcc_2d = X_val_mfcc.reshape(
    -1,
    X_val_mfcc.shape[-1]
)

X_test_mfcc_2d = X_test_mfcc.reshape(
    -1,
    X_test_mfcc.shape[-1]
)

scaler_mfcc.fit(
    X_train_mfcc_2d
)

X_train_mfcc_2d = scaler_mfcc.transform(
    X_train_mfcc_2d
)

X_val_mfcc_2d = scaler_mfcc.transform(
    X_val_mfcc_2d
)

X_test_mfcc_2d = scaler_mfcc.transform(
    X_test_mfcc_2d
)

X_train_mfcc = X_train_mfcc_2d.reshape(
    X_train_mfcc.shape
)

X_val_mfcc = X_val_mfcc_2d.reshape(
    X_val_mfcc.shape
)

X_test_mfcc = X_test_mfcc_2d.reshape(
    X_test_mfcc.shape
)

# ==========================================
# MEL LOG TRANSFORM
# ==========================================

X_train_mel = np.log1p(
    X_train_mel
)

X_val_mel = np.log1p(
    X_val_mel
)

X_test_mel = np.log1p(
    X_test_mel
)


# ==========================================
# MEL SCALING
# ==========================================

scaler_mel = StandardScaler()

X_train_mel_2d = X_train_mel.reshape(
    -1,
    X_train_mel.shape[-1]
)

X_val_mel_2d = X_val_mel.reshape(
    -1,
    X_val_mel.shape[-1]
)

X_test_mel_2d = X_test_mel.reshape(
    -1,
    X_test_mel.shape[-1]
)

scaler_mel.fit(
    X_train_mel_2d
)

X_train_mel_2d = scaler_mel.transform(
    X_train_mel_2d
)

X_val_mel_2d = scaler_mel.transform(
    X_val_mel_2d
)

X_test_mel_2d = scaler_mel.transform(
    X_test_mel_2d
)

X_train_mel = X_train_mel_2d.reshape(
    X_train_mel.shape
)

X_val_mel = X_val_mel_2d.reshape(
    X_val_mel.shape
)

X_test_mel = X_test_mel_2d.reshape(
    X_test_mel.shape
)
# print("\nAfter Mel Scaling")
# print(np.mean(X_train_mel))
# print(np.std(X_train_mel))
# print(np.min(X_train_mel))
# print(np.max(X_train_mel))

joblib.dump(
    scaler_mfcc,
    MFCC_SCALER_PATH
)

joblib.dump(
    scaler_mel,
    MEL_SCALER_PATH
)
# print("\nMFCC Stats")
# print("Mean :", np.mean(X_train_mfcc))
# print("Std  :", np.std(X_train_mfcc))
# print("Min  :", np.min(X_train_mfcc))
# print("Max  :", np.max(X_train_mfcc))

# print("\nMel Stats")
# print("Mean :", np.mean(X_train_mel))
# print("Std  :", np.std(X_train_mel))
# print("Min  :", np.min(X_train_mel))
# print("Max  :", np.max(X_train_mel))

# print("\nNaN Check")
# print("MFCC NaN :", np.isnan(X_train_mfcc).sum())
# print("Mel NaN  :", np.isnan(X_train_mel).sum())

# print("\nInf Check")
# print("MFCC Inf :", np.isinf(X_train_mfcc).sum())
# print("Mel Inf  :", np.isinf(X_train_mel).sum())

# mel_flat = X_train_mel.reshape(-1)

# print("\nMel Percentiles")
# print("50% :", np.percentile(mel_flat, 50))
# print("90% :", np.percentile(mel_flat, 90))
# print("95% :", np.percentile(mel_flat, 95))
# print("99% :", np.percentile(mel_flat, 99))
# print("99.9% :", np.percentile(mel_flat, 99.9))
# print("Max :", np.max(mel_flat))
"""
Saving the original label
"""
y_train_labels = y_train.copy()

# ==========================================
# ONE HOT ENCODING
# ==========================================

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

"""
    Class Weights
"""
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train_labels),
    y=y_train_labels
)

class_weights = dict(
    enumerate(class_weights)
)

print("\nClass Weights:")
print(class_weights)


# print("\ny_train:", y_train.shape)
# print("y_val:", y_val.shape)
# print("y_test:", y_test.shape)

# ==========================================
# MEL CHANNEL DIMENSION
# ==========================================

X_train_mel = X_train_mel[..., np.newaxis]

X_val_mel = X_val_mel[..., np.newaxis]

X_test_mel = X_test_mel[..., np.newaxis]

# print("\nMFCC Shape:", X_train_mfcc.shape)

# print("Mel Shape:", X_train_mel.shape)

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
            initializer="glorot_uniform",
            trainable=True
        )

        self.b = self.add_weight(
            name="attention_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True
        )

        super().build(
            input_shape
        )

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
# MFCC BRANCH
# ==========================================

mfcc_input = Input(
    shape=(120, 130),
    name="mfcc_input"
)

mfcc_x = Bidirectional(
    LSTM(
        64,
        return_sequences=True
    )
)(mfcc_input)

mfcc_x = Dropout(
    LSTM_DROPOUT
)(mfcc_x)

mfcc_x = Bidirectional(
    LSTM(
        64,
        return_sequences=True
    )
)(mfcc_x)

mfcc_output = AttentionLayer()(
    mfcc_x
)

mfcc_output = Dropout(
    0.2
)(
    mfcc_output
)

print("\nMFCC Branch Ready")
# ==========================================
# MEL BRANCH
# ==========================================

mel_input = Input(
    shape=(128, 130, 1),
    name="mel_input"
)

# Block 1

mel_x = Conv2D(
    32,
    (3, 3),
    padding="same",
    kernel_regularizer=l2(
        L2_VALUE
    )
)(mel_input)

mel_x = BatchNormalization()(
    mel_x
)

mel_x = ReLU()(
    mel_x
)

mel_x = MaxPooling2D(
    pool_size=(2, 2)
)(
    mel_x
)

# Block 2

mel_x = Conv2D(
    64,
    (3, 3),
    padding="same",
    kernel_regularizer=l2(
        L2_VALUE
    )
)(
    mel_x
)

mel_x = BatchNormalization()(
    mel_x
)

mel_x = ReLU()(
    mel_x
)

mel_x = MaxPooling2D(
    pool_size=(2, 2)
)(
    mel_x
)

# Block 3

mel_x = Conv2D(
    128,
    (3, 3),
    padding="same",
    kernel_regularizer=l2(
        L2_VALUE
    )
)(
    mel_x
)

mel_x = BatchNormalization()(
    mel_x
)

mel_x = ReLU()(
    mel_x
)

# Global Pooling

mel_output = GlobalAveragePooling2D()(
    mel_x
)

print("\nMel Branch Ready")

# ==========================================
# FUSION
# ==========================================

fusion = Concatenate()(
    [
        mfcc_output,
        mel_output
    ]
)

fusion = Dropout(
    FUSION_DROPOUT_1
)(
    fusion
)

fusion = Dense(
    128
)(
    fusion
)

fusion = BatchNormalization()(
    fusion
)

fusion = ReLU()(
    fusion
)

fusion = Dropout(
    FUSION_DROPOUT_2
)(
    fusion
)

fusion = Dense(
    64
)(
    fusion
)
fusion = BatchNormalization()(
    fusion
)

fusion = ReLU()(
    fusion
)

outputs = Dense(
    NUM_CLASSES,
    activation="softmax"
)(
    fusion
)
# ==========================================
# BUILD MODEL
# ==========================================

model = Model(
    inputs=[
        mfcc_input,
        mel_input
    ],
    outputs=outputs
)

model.summary()
print(model.input_shape)
print(model.output_shape)

# ==========================================
# COMPILE MODEL
# ==========================================

optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE
)

model.compile(
    optimizer=optimizer,
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel Compiled Successfully")


# ==========================================
# CREATE MODELS FOLDER
# ==========================================

# import os

# os.makedirs(
#     "models",
#     exist_ok=True
# )

# ==========================================
# CALLBACKS
# ==========================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=EARLY_STOPPING_PATIENCE,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=REDUCE_LR_FACTOR,
    patience=REDUCE_LR_PATIENCE,
    min_lr=MIN_LR,
    verbose=1
)

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

print("\nCallbacks Ready")

# ==========================================
# TRAINING
# ==========================================
history = model.fit(
    x=[
        X_train_mfcc,
        X_train_mel
    ],
    y=y_train,
    validation_data=(
        [
            X_val_mfcc,
            X_val_mel
        ],
        y_val
    ),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weights,
    callbacks=[
        early_stopping,
        reduce_lr,
        checkpoint
    ]
)