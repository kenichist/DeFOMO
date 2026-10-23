import pandas as pd

# change path if needed
df = pd.read_csv("train.tsv", sep="\t")

# keep only rows that have an image URL and some text
df = df.dropna(subset=["image_url", "clean_title"])

# choose how many samples you want
N_TRAIN = 10000  # adjust based on your 10GB estimate

df_small = df.sample(n=N_TRAIN, random_state=42)
df_small.to_csv("train_small.tsv", sep="\t", index=False)

print("Original size:", len(df), "→ subset size:", len(df_small))
