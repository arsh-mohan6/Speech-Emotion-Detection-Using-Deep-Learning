# ==========================================
# IMPORTS
# ==========================================

import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler

# ==========================================
# LOAD TRAIN FEATURES
# ==========================================

X_train = np.load(
    "data/processed_data/X_train.npy"
)

print("Original Shape:", X_train.shape)

# ==========================================
# RESHAPE FOR SCALER
# ==========================================

X_train_2d = X_train.reshape(
    -1,
    X_train.shape[-1]
)

print("Scaler Shape:", X_train_2d.shape)

# ==========================================
# FIT SCALER
# ==========================================

scaler = StandardScaler()

scaler.fit(
    X_train_2d
)

# ==========================================
# SAVE SCALER
# ==========================================

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

print("Scaler saved successfully.")