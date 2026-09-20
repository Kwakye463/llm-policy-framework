
# load_dataset.py
#
# One-time script: downloads the deepset/prompt-injections dataset (Apache-2.0, public) and saves it locally as CSV files so the rest
# of the project never needs the `datasets` library or an internet connection again.
#
# Dataset: https://huggingface.co/datasets/deepset/prompt-injections
# Columns: text (the prompt), label (1 = injection, 0 = legitimate)
 
import pandas as pd
 
BASE = "hf://datasets/deepset/prompt-injections/"
SPLITS = {
    "train": "data/train-00000-of-00001-9564e8b05b4757ab.parquet",
    "test":  "data/test-00000-of-00001-701d16158af87368.parquet",
}
 
print("Downloading deepset/prompt-injections ...")
 
train_df = pd.read_parquet(BASE + SPLITS["train"])
test_df  = pd.read_parquet(BASE + SPLITS["test"])
 
# Save locally as CSV (readable by pandas with no extra dependencies)
train_df.to_csv("deepset_train.csv", index=False)
test_df.to_csv("deepset_test.csv", index=False)
 
# Report what we got
print("\nSaved deepset_train.csv and deepset_test.csv\n")
print("TRAIN split:")
print("  rows   :", len(train_df))
print("  columns:", list(train_df.columns))
print("  labels :", train_df["label"].value_counts().to_dict())
print("\nTEST split:")
print("  rows   :", len(test_df))
print("  labels :", test_df["label"].value_counts().to_dict())
 
print("\nFirst 3 training rows:")
for _, row in train_df.head(3).iterrows():
    label = "INJECTION" if row["label"] == 1 else "legitimate"
    print(f"  [{label}] {str(row['text'])[:70]}")
    