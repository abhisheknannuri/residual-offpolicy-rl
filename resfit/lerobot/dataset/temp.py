import pandas as pd
from datasets import load_dataset

HF_DATASET = "poolvarine/robomimic-mh-lift-image-dense-jackpot-scaled-50"

print("Loading dataset from cache...")
ds = load_dataset(
    "parquet", 
    data_files=f"hf://datasets/{HF_DATASET}/**/*.parquet", 
    split="train"
)

df = ds.to_pandas()

print(f"Header columns: {df.columns.tolist()}")

# Save a clean copy right next to your script!
export_path = "robomimic_preview_dense.parquet"
df.to_parquet(export_path)
print(f"✅ Saved local copy for viewing: {export_path}")