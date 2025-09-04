import pandas as pd, base64, mimetypes
from pathlib import Path
from typing import Optional

CSV_PATH = Path("INVENTORY/TESTRUNIMG_Food Partners Database - Inventory.csv")
PARQUET_PATH = Path("INVENTORY/dataset.parquet")

IMG_COLS = [
    ("Store Image", "store_image_bytes", "store_image_mimetype"),
    ("Food Image 1", "food_image_1_bytes", "food_image_1_mimetype"),
    ("Food Image 2", "food_image_2_bytes", "food_image_2_mimetype"),
    ("Food Image 3", "food_image_3_bytes", "food_image_3_mimetype"),
]

def read_as_bytes(p: Optional[str]):
    if not isinstance(p, str):
        return None
    p = p.strip()
    if not p:
        return None
    # Support data URLs
    if p.startswith("data:"):
        try:
            _, b64 = p.split(",", 1)
            return base64.b64decode(b64)
        except Exception:
            return None
    # Treat as file path
    path = Path(p)
    if not path.exists():
        # Try relative to CSV dir
        alt = CSV_PATH.parent / p
        if alt.exists():
            return alt.read_bytes()
        return None
    return path.read_bytes()

def guess_mime(p: Optional[str]):
    if not isinstance(p, str):
        return None
    p = p.strip()
    if not p:
        return None
    if p.startswith("data:"):
        try:
            head = p.split(",", 1)[0]
            return head.split(":")[1].split(";")[0]
        except Exception:
            return None
    mt, _ = mimetypes.guess_type(p)
    return mt

def main():
    df = pd.read_csv(CSV_PATH)
    # Add id if missing
    if "id" not in df.columns:
        df.insert(0, "id", range(1, len(df) + 1))

    for src, out_bytes, out_mime in IMG_COLS:
        if src in df.columns:
            df[out_bytes] = df[src].apply(read_as_bytes)
            df[out_mime] = df[src].apply(guess_mime)
        else:
            df[out_bytes] = None
            df[out_mime] = None

    df.to_parquet(PARQUET_PATH, index=False, engine="pyarrow")

    # Report how many images actually loaded
    counts = {src: int(df[out_bytes].apply(lambda x: isinstance(x, (bytes, bytearray))).sum())
              for src, out_bytes, _ in IMG_COLS}
    print("Wrote", PARQUET_PATH)
    print("Images found:", counts)

if __name__ == "__main__":
    main()