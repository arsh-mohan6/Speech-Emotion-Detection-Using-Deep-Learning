import pandas as pd
from sklearn.model_selection import train_test_split

# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(
    "data/processed_data/combined_dataset.csv"
)

print("Total Samples:", len(df))

# ==========================================
# SPLIT EACH DATASET SEPARATELY
# ==========================================

train_list = []
val_list = []
test_list = []

for dataset_name in [
    "RAVDESS",
    "CREMA-D",
    "TESS"
]:

    dataset_df = df[
        df["dataset"] == dataset_name
    ]

    print(
        f"\n{dataset_name}:",
        len(dataset_df)
    )

    # 80% Train
    train_df, temp_df = train_test_split(
        dataset_df,
        test_size=0.20,
        stratify=dataset_df["emotion"],
        random_state=42
    )

    # 10% Val
    # 10% Test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["emotion"],
        random_state=42
    )

    train_list.append(train_df)
    val_list.append(val_df)
    test_list.append(test_df)

# ==========================================
# COMBINE SPLITS
# ==========================================

train_df = pd.concat(
    train_list,
    ignore_index=True
)

val_df = pd.concat(
    val_list,
    ignore_index=True
)

test_df = pd.concat(
    test_list,
    ignore_index=True
)

# Shuffle
train_df = train_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

val_df = val_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

test_df = test_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# ==========================================
# SAVE FILES
# ==========================================

train_df.to_csv(
    "data/processed_data/train.csv",
    index=False
)

val_df.to_csv(
    "data/processed_data/val.csv",
    index=False
)

test_df.to_csv(
    "data/processed_data/test.csv",
    index=False
)

# ==========================================
# PRINT STATS
# ==========================================

print("\nTrain:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

print("\nTrain Dataset Distribution:")
print(
    train_df["dataset"].value_counts()
)

print("\nValidation Dataset Distribution:")
print(
    val_df["dataset"].value_counts()
)

print("\nTest Dataset Distribution:")
print(
    test_df["dataset"].value_counts()
)

print("\nTrain Emotion Distribution:")
print(
    train_df["emotion"].value_counts()
)