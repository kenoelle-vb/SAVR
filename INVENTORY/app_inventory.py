from flask import Flask, send_file, abort, request, make_response
import pandas as pd
from io import BytesIO
from pathlib import Path
from PIL import Image, ImageOps

PARQUET_PATH = Path("INVENTORY/dataset.parquet")

# Map query 'col' values to the Parquet binary/mimetype columns
COL_MAP = {
    "store": ("store_image_bytes", "store_image_mimetype"),
    "food1": ("food_image_1_bytes", "food_image_1_mimetype"),
    "food2": ("food_image_2_bytes", "food_image_2_mimetype"),
    "food3": ("food_image_3_bytes", "food_image_3_mimetype"),
}

# Reasonable safety caps (avoid huge resize requests)
MAX_W = 4096
MAX_H = 4096

app = Flask(__name__)

# Load Parquet once at startup (simple and fine for small/medium datasets)
df = pd.read_parquet(PARQUET_PATH, engine="pyarrow")
if "id" not in df.columns:
    raise RuntimeError("Parquet must contain an 'id' column.")
df.set_index("id", inplace=True, drop=False)

def get_binary_and_mime(row, which: str):
    """Return (bytes, mimetype) for the requested column key; None if missing."""
    bytes_col, mime_col = COL_MAP[which]
    data = row.get(bytes_col)
    # Some engines may materialize as memoryview
    if isinstance(data, memoryview):
        data = data.tobytes()
    if not isinstance(data, (bytes, bytearray)):
        return None, None
    mime = (row.get(mime_col) or "").lower() or "image/jpeg"
    return data, mime

def parse_int(name, default=None, minimum=1, maximum=None):
    v = request.args.get(name, default)
    if v is None:
        return None
    try:
        v = int(v)
        if v < minimum:
            return None
        if maximum is not None and v > maximum:
            v = maximum
        return v
    except Exception:
        return None

def choose_format(requested_fmt: str | None, src_mime: str, pil_mode: str) -> tuple[str, str]:
    """
    Decide output (Pillow format name, response mimetype).
    If requested, honor it; otherwise JPEG unless alpha present -> PNG.
    """
    if requested_fmt:
        f = requested_fmt.lower()
        if f == "jpeg" or f == "jpg":
            return ("JPEG", "image/jpeg")
        if f == "png":
            return ("PNG", "image/png")
        if f == "webp":
            return ("WEBP", "image/webp")
    # No explicit fmt: if source appears to have alpha (RGBA/LA), prefer PNG
    if "A" in pil_mode:  # has alpha channel
        return ("PNG", "image/png")
    # default
    return ("JPEG", "image/jpeg")

def resize_bytes(data: bytes, w: int | None, h: int | None, mode: str, out_fmt: str | None, quality: int | None, src_mime: str):
    """
    Resize according to mode: fit (contain), cover (fill+crop), pad (letterbox).
    Returns (bytes_io, response_mimetype).
    """
    # If no size requested, just stream original bytes with original mimetype
    if not w and not h:
        return BytesIO(data), src_mime

    # Clamp/normalize target box
    W = parse_int("w", None, minimum=1, maximum=MAX_W) if w is None else w
    H = parse_int("h", None, minimum=1, maximum=MAX_H) if h is None else h
    if W is None or H is None:
        abort(400, "Both w and h must be positive integers.")

    # Open with Pillow
    im = Image.open(BytesIO(data))
    # Convert palette etc to a sensible base
    if im.mode in ("P", "CMYK"):
        im = im.convert("RGBA")
    pil_mode = im.mode

    mode = (mode or "fit").lower()
    if mode not in ("fit", "cover", "pad"):
        abort(400, "mode must be one of: fit, cover, pad")

    if mode == "fit":
        # Contain within box, preserve aspect, <= W x H
        im_copy = im.copy()
        im_copy.thumbnail((W, H))  # in-place, preserves aspect
        out_im = im_copy

    elif mode == "cover":
        # Fill the box and center-crop to exact W x H
        out_im = ImageOps.fit(im, (W, H), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))

    else:  # pad (letterbox)
        # Pads image to exactly W x H with black background (can change color if desired)
        # For transparent sources, keep transparency; otherwise pad with black
        if "A" in im.mode:
            out_im = ImageOps.pad(im, (W, H), method=Image.Resampling.LANCZOS, color=None, centering=(0.5, 0.5))
        else:
            out_im = ImageOps.pad(im, (W, H), method=Image.Resampling.LANCZOS, color=(0, 0, 0), centering=(0.5, 0.5))

    # Decide output format & mimetype
    fmt, resp_mime = choose_format(out_fmt, src_mime, out_im.mode)

    # JPEG cannot store alpha; if chosen, drop alpha safely
    if fmt == "JPEG" and ("A" in out_im.mode):
        out_im = out_im.convert("RGB")

    # Encode
    out = BytesIO()
    save_kwargs = {}
    if fmt == "JPEG":
        save_kwargs["quality"] = quality if (quality and 1 <= quality <= 95) else 85
        save_kwargs["optimize"] = True
        save_kwargs["progressive"] = True
    elif fmt == "WEBP":
        save_kwargs["quality"] = quality if (quality and 1 <= quality <= 95) else 85
        save_kwargs["method"] = 6
    out_im.save(out, format=fmt, **save_kwargs)
    out.seek(0)
    return out, resp_mime

@app.get("/img/<int:item_id>")
def serve_image(item_id: int):
    which = request.args.get("col", "store")
    if which not in COL_MAP:
        abort(400, f"Invalid col. Use one of: {', '.join(COL_MAP)}")
    if item_id not in df.index:
        abort(404, "Record not found.")

    row = df.loc[item_id]
    data, src_mime = get_binary_and_mime(row, which)
    if data is None:
        abort(404, f"No image available for id={item_id}, col={which}")

    # Parse resize/output options
    w = parse_int("w", None, minimum=1, maximum=MAX_W)
    h = parse_int("h", None, minimum=1, maximum=MAX_H)
    mode = request.args.get("mode", "fit")
    fmt = request.args.get("fmt")  # jpeg|png|webp
    q = parse_int("q", None, minimum=1, maximum=95)

    buf, resp_mime = resize_bytes(data, w, h, mode, fmt, q, src_mime)

    # Add basic cache headers (optional)
    resp = make_response(send_file(buf, mimetype=resp_mime, download_name=f"{item_id}-{which}"))
    resp.headers["Cache-Control"] = "public, max-age=3600"
    return resp

@app.get("/")
def index():
    # Show first 20 records, choosing the first available image among store/food1/food2/food3
    ids = df["id"].head(20).tolist()
    blocks = []
    for i in ids:
        row = df.loc[i]
        which = "store"
        for candidate in ["store", "food1", "food2", "food3"]:
            data, _ = get_binary_and_mime(row, candidate)
            if data:
                which = candidate
                break
        blocks.append(
            f'<div style="text-align:center">'
            f'<img src="/img/{i}?col={which}&w=360&h=640&mode=fit" '
            f'style="max-height:160px;display:block;margin:auto">'
            f'<div>id {i} ({which})</div>'
            f'</div>'
        )
    return (
        """<!doctype html>
        <meta charset="utf-8">
        <title>Gallery</title>
        <h1>Gallery (360×640 fit previews)</h1>
        <div style='display:flex;gap:12px;flex-wrap:wrap'>"""
        + "\n".join(blocks)
        + "</div>"
    )

@app.get("/health")
def health():
    return {"status": "ok", "rows": int(len(df))}

if __name__ == "__main__":
    app.run(debug=True)