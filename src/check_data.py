# import numpy as np

# X_train = np.load("data/processed_data/X_train.npy")
# y_train = np.load("data/processed_data/y_train.npy")

# X_val = np.load("data/processed_data/X_val.npy")
# y_val = np.load("data/processed_data/y_val.npy")

# X_test = np.load("data/processed_data/X_test.npy")
# y_test = np.load("data/processed_data/y_test.npy")

# print("Train:", X_train.shape, y_train.shape)
# print("Val:", X_val.shape, y_val.shape)
# print("Test:", X_test.shape, y_test.shape)

# print("Train dtype:", X_train.dtype)
# print("Labels dtype:", y_train.dtype)

import tensorflow as tf

print(tf.config.list_physical_devices())
print(tf.config.list_physical_devices('GPU'))

import numpy as np

X_train = np.load("data/processed_data/X_train.npy")

print("Train Min :", X_train[0].min())
print("Train Max :", X_train[0].max())
print("Train Mean:", X_train[0].mean())