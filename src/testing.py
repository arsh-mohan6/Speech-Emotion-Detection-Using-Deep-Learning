import numpy as np

X_train = np.load(
    "data/processed_data/X_train.npy"
)

print(X_train.mean())
print(X_train.std())