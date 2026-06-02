import pandas as pd
from sklearn.model_selection import train_test_split

# Load combined dataset
df = pd.read_csv("data/processed_data/combined_dataset.csv")

print("Total Samples:", len(df))

# First split: Train (80%) and Temp (20%)
train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    stratify=df["emotion"],
    random_state=42
)

# Second split: Validation (10%) and Test (10%)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["emotion"],
    random_state=42
)

# Save files
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

# Print stats
print("\nTrain:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))

print("\nTrain Emotion Distribution:")
print(train_df["emotion"].value_counts())