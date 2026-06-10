# ==========================================
# TRAINING CONFIG
# ==========================================

NUM_CLASSES = 6

BATCH_SIZE = 64

EPOCHS = 50

LEARNING_RATE = 0.001


# ==========================================
# DROPOUT
# ==========================================

LSTM_DROPOUT = 0.3

FUSION_DROPOUT_1 = 0.3

FUSION_DROPOUT_2 = 0.3


# ==========================================
# REGULARIZATION
# ==========================================

USE_L2 = True

L2_VALUE = 1e-4


# ==========================================
# CALLBACKS
# ==========================================

EARLY_STOPPING_PATIENCE = 10

REDUCE_LR_PATIENCE = 3

REDUCE_LR_FACTOR = 0.5

MIN_LR = 1e-6


# ==========================================
# MODEL PATHS
# ==========================================

MODEL_PATH = "models/best_model.keras"

MFCC_SCALER_PATH = "models/scaler_mfcc.pkl"

MEL_SCALER_PATH = "models/scaler_mel.pkl"