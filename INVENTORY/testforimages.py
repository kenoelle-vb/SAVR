from pathlib import Path
csv_file = Path("INVENTORY/TESTRUNIMG_Food Partners Database - Inventory.csv")

# Pick a row you expect has Food Image 2
import pandas as pd
df = pd.read_csv(csv_file)
print(df["Food Image 2"].head())   # show first few paths

# Now check existence
for p in df["Food Image 2"].dropna().head():
    candidate = Path(p)
    alt = csv_file.parent / p
    print(p, "exists as-is?", candidate.exists(), "exists relative?", alt.exists())
