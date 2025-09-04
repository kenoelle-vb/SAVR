import os
import json
from io import BytesIO
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path

import pandas as pd
from flask import (
    Flask, render_template, request, redirect, url_for, session, g, abort, send_file, make_response, flash
)

# Accounts DB (Google Sheets)
import gspread

# Geolocation fallback + image resizing
import geocoder
from PIL import Image, ImageOps

# -----------------------------
# Flask config
# -----------------------------
app = Flask(
    __name__,
    template_folder="templates",
    static_folder="C:/Users/keno/OneDrive/Documents/Projects/SAVR App"
)
app.secret_key = os.urandom(24)

email_logged = None  # simple single-user session pattern

# -----------------------------
# Accounts DB (Google Sheets)
# -----------------------------
# Replaced the direct filename with a placeholder for the secret
GOOGLE_SA_JSON_SECRET_NAME = "SAVR_JSON_KEY"

def _get_google_creds_file() -> str | None:
    """
    Fetches the Google service account JSON from a GitHub secret,
    writes it to a temporary file, and returns the filename.
    """
    json_str = os.environ.get(GOOGLE_SA_JSON_SECRET_NAME)
    if not json_str:
        print(f"Environment variable {GOOGLE_SA_JSON_SECRET_NAME} not found.")
        return None
    try:
        creds_data = json.loads(json_str)
        temp_file_path = "google_creds.json"
        with open(temp_file_path, "w") as temp_file:
            json.dump(creds_data, temp_file)
        print(f"Credentials written to temporary file: {temp_file_path}")
        return temp_file_path
    except Exception as e:
        print(f"Error processing JSON secret: {e}")
        return None

ACCOUNTS_SHEET_KEY = "1cMG-4L5SfkbQsNze3u8mnXvdgHh5MeQqdB4NLu-0qmM"

def _worksheet_to_df(ws):
    """Convert a gspread Worksheet to a pandas DataFrame (keeps the first row as header)."""
    values = ws.get_all_values()
    if not values:
        return pd.DataFrame()
    header, rows = values[0], values[1:]
    return pd.DataFrame(rows, columns=[c.strip() for c in header])

def get_db():
    """
    Load 'User' and 'Partner' tables from Google Sheets via gspread and cache in Flask's g.
    No XLSX/CSV download; this reads the live source.
    """
    if 'df_user' not in g or 'df_partner' not in g:
        try:
            creds_file = _get_google_creds_file()
            if not creds_file:
                raise FileNotFoundError("Google credentials file could not be created.")
            gc = gspread.service_account(filename=creds_file)
            sh = gc.open_by_key(ACCOUNTS_SHEET_KEY)
            ws_user = sh.worksheet("User")
            ws_partner = sh.worksheet("Partner")
            g.df_user = _worksheet_to_df(ws_user)
            g.df_partner = _worksheet_to_df(ws_partner)
        except Exception as e:
            print(f"[get_db] Error opening Google Sheet: {e}")
            g.df_user = pd.DataFrame()
            g.df_partner = pd.DataFrame()
    return g.df_user, g.df_partner

def add_user_to_db(first_name, last_name, email, password):
    """Append a new row to the 'User' worksheet."""
    try:
        creds_file = _get_google_creds_file()
        if not creds_file:
            raise FileNotFoundError("Google credentials file could not be created.")
        gc = gspread.service_account(filename=creds_file)
        sh = gc.open_by_key(ACCOUNTS_SHEET_KEY)
        ws = sh.worksheet("User")
        ws.append_row([first_name, last_name, email, password, "User"])
    except Exception as e:
        print(f"[add_user_to_db] Error: {e}")

def add_partner_to_db(partner_name, owner_name, nik_ktp, email, password):
    """Append a new row to the 'Partner' worksheet."""
    try:
        creds_file = _get_google_creds_file()
        if not creds_file:
            raise FileNotFoundError("Google credentials file could not be created.")
        gc = gspread.service_account(filename=creds_file)
        sh = gc.open_by_key(ACCOUNTS_SHEET_KEY)
        ws = sh.worksheet("Partner")
        ws.append_row([partner_name, owner_name, nik_ktp, email, password, "Partner"])
    except Exception as e:
        print(f"[add_partner_to_db] Error: {e}")

def find_account_by_email(email: str):
    """
    Look up an account by email across both sheets. Returns dict or None:
    { 'name': '...', 'email': '...', 'role': 'User'|'Partner' }
    """
    df_user, df_partner = get_db()
    email_l = (email or "").strip().lower()

    # Search User
    if not df_user.empty and "E-mail" in df_user.columns:
        m = df_user["E-mail"].str.strip().str.lower() == email_l
        if m.any():
            row = df_user[m].iloc[0]
            first = (row.get("First Name") or "").strip()
            last = (row.get("Last Name") or "").strip()
            name = (first + " " + last).strip() or first or last or "User"
            role = (row.get("Role") or "User").strip() or "User"
            return {"name": name, "email": email, "role": role}

    # Search Partner
    if not df_partner.empty and "E-mail" in df_partner.columns:
        m = df_partner["E-mail"].str.strip().str.lower() == email_l
        if m.any():
            row = df_partner[m].iloc[0]
            name = (row.get("Partner Name") or row.get("Owner Name") or "Partner").strip()
            role = (row.get("Role") or "Partner").strip() or "Partner"
            return {"name": name, "email": email, "role": role}

    return None

# -----------------------------
# Distance helpers
# -----------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return round(R * c, 1)

DEFAULT_CITY_LAT = -6.9222
DEFAULT_CITY_LON = 107.6069

def get_user_location(lat_param: str | None, lon_param: str | None):
    try:
        if lat_param and lon_param:
            return float(lat_param), float(lon_param)
    except Exception:
        pass
    if session.get("fallback_lat") is not None and session.get("fallback_lon") is not None:
        return float(session["fallback_lat"]), float(session["fallback_lon"])
    try:
        g_ip = geocoder.ip('me')
        if g_ip and g_ip.latlng:
            lat, lon = float(g_ip.latlng[0]), float(g_ip.latlng[1])
            session["fallback_lat"], session["fallback_lon"] = lat, lon
            return lat, lon
    except Exception as e:
        print("geocoder.ip fallback failed:", e)
    session["fallback_lat"], session["fallback_lon"] = DEFAULT_CITY_LAT, DEFAULT_CITY_LON
    return DEFAULT_CITY_LAT, DEFAULT_CITY_LON

# -----------------------------
# Parquet-backed image dataset
# -----------------------------
PARQUET_PATH = Path("INVENTORY/dataset.parquet")
parquet_df = None
store_to_pid: dict[str, int] = {}
food_slot_map: dict[tuple[str, str], str] = {}

COL_MAP = {
    "store": ("store_image_bytes", "store_image_mimetype"),
    "food1": ("food_image_1_bytes", "food_image_1_mimetype"),
    "food2": ("food_image_2_bytes", "food_image_2_mimetype"),
    "food3": ("food_image_3_bytes", "food_image_3_mimetype"),
}

def _bytes_present(val) -> bool:
    if isinstance(val, (bytes, bytearray)):
        return True
    try:
        return isinstance(val, memoryview) and len(val) > 0
    except Exception:
        return False

def init_parquet():
    global parquet_df, store_to_pid, food_slot_map
    if not PARQUET_PATH.exists():
        parquet_df = None; store_to_pid = {}; food_slot_map = {}; return
    parquet_df = pd.read_parquet(PARQUET_PATH, engine="pyarrow").copy()
    if "id" not in parquet_df.columns:
        parquet_df.insert(0, "id", range(1, len(parquet_df) + 1))
    parquet_df.set_index("id", inplace=True)
    store_to_pid = {}
    if "Store Name" in parquet_df.columns:
        for pid, row in parquet_df.iterrows():
            name = str(row.get("Store Name", "")).strip()
            if name:
                store_to_pid[name] = int(pid)
    food_slot_map = {}
    for pid, row in parquet_df.iterrows():
        store_name = str(row.get("Store Name", "")).strip()
        if not store_name: continue
        for slot_num in (1, 2, 3):
            fname_col = f"Food Name {slot_num}"
            if fname_col in parquet_df.columns:
                fname = row.get(fname_col)
                if isinstance(fname, str) and fname.strip():
                    food_slot_map[(store_name, fname.strip())] = f"food{slot_num}"

init_parquet()

# -----------------------------
# Image serving
# -----------------------------
def _choose_format(requested_fmt: str | None, pil_mode: str) -> tuple[str, str]:
    if requested_fmt:
        f = requested_fmt.lower()
        if f in ("jpeg", "jpg"): return ("JPEG", "image/jpeg")
        if f == "png": return ("PNG", "image/png")
        if f == "webp": return ("WEBP", "image/webp")
    if "A" in pil_mode: return ("PNG", "image/png")
    return ("JPEG", "image/jpeg")

def _parse_int(name, default=None, minimum=1, maximum=None):
    v = request.args.get(name, default)
    if v is None: return None
    try:
        x = int(v)
        if x < minimum: return None
        if maximum is not None and x > maximum: x = maximum
        return x
    except Exception:
        return None

@app.get("/img/<int:item_id>")
def serve_image(item_id: int):
    if parquet_df is None: abort(404)
    col_key = request.args.get("col", "store")
    if col_key not in COL_MAP: abort(400, f"Invalid col")
    try:
        row = parquet_df.loc[item_id]
    except KeyError:
        abort(404)
    bytes_col, mime_col = COL_MAP[col_key]
    data = row.get(bytes_col)
    if isinstance(data, memoryview): data = data.tobytes()
    if not isinstance(data, (bytes, bytearray)): abort(404)
    src_mime = (row.get(mime_col) or "image/jpeg").lower()

    MAX_W, MAX_H = 4096, 4096
    w = _parse_int("w", None, 1, MAX_W)
    h = _parse_int("h", None, 1, MAX_H)
    mode = (request.args.get("mode", "fit") or "fit").lower()
    fmt_req = request.args.get("fmt")
    q = _parse_int("q", None, 1, 95)

    if not w and not h:
        return send_file(BytesIO(data), mimetype=src_mime, download_name=f"{item_id}-{col_key}")

    if mode not in ("fit", "cover", "pad"): abort(400)
    if not (w and h): abort(400, "need w and h")

    im = Image.open(BytesIO(data))
    if im.mode in ("P", "CMYK"): im = im.convert("RGBA")
    if mode == "fit":
        im_copy = im.copy(); im_copy.thumbnail((w, h)); out_im = im_copy
    elif mode == "cover":
        out_im = ImageOps.fit(im, (w, h), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    else:
        if "A" in im.mode:
            out_im = ImageOps.pad(im, (w, h), method=Image.Resampling.LANCZOS, color=None, centering=(0.5, 0.5))
        else:
            out_im = ImageOps.pad(im, (w, h), method=Image.Resampling.LANCZOS, color=(0,0,0), centering=(0.5,0.5))

    fmt, resp_mime = _choose_format(fmt_req, out_im.mode)
    if fmt == "JPEG" and "A" in out_im.mode: out_im = out_im.convert("RGB")
    out = BytesIO()
    save_kwargs = {}
    if fmt in ("JPEG","WEBP"):
        save_kwargs["quality"] = q if (q and 1 <= q <= 95) else 85
        if fmt == "JPEG": save_kwargs.update(optimize=True, progressive=True)
        if fmt == "WEBP": save_kwargs["method"] = 6
    out_im.save(out, format=fmt, **save_kwargs); out.seek(0)
    resp = make_response(send_file(out, mimetype=resp_mime, download_name=f"{item_id}-{col_key}"))
    resp.headers["Cache-Control"] = "public, max-age=3600"
    return resp

# -----------------------------
# Business data (Excel) + Parquet image URLs
# -----------------------------
def load_food_data():
    file_path = 'EXAMPLE RUN_Food Partners Database.xlsx'
    if not os.path.exists(file_path): return [], None, None
    try:
        df_eng = pd.read_excel(file_path, sheet_name='Engagement')
        df_inv = pd.read_excel(file_path, sheet_name='Inventory')
    except Exception as e:
        print("Error loading Excel:", e); return [], None, None

    stores = []
    if not df_eng.empty and not df_inv.empty:
        for idx, row in df_eng.iterrows():
            store_name = row['Store Name']
            lat = float(row['Store Latitude']); lon = float(row['Store Longitude'])
            external_img = (row['Store Image'].replace('imgur.com/','i.imgur.com/') + '.jpg') if pd.notna(row['Store Image']) else ''
            img_url = external_img
            pid = store_to_pid.get(store_name)
            if parquet_df is not None and pid:
                rowp = parquet_df.loc[pid]
                slot_choice = None
                if _bytes_present(rowp.get(COL_MAP["store"][0])): slot_choice = "store"
                else:
                    for s in ("food1","food2","food3"):
                        if _bytes_present(rowp.get(COL_MAP[s][0])): slot_choice = s; break
                if slot_choice:
                    img_url = url_for("serve_image", item_id=pid, col=slot_choice, w=360, h=640, mode="fit")

            stores.append({
                'name': store_name, 'lat': lat, 'lon': lon, 'image': img_url,
                'engagement': row.get('Store Engagement', 0) if pd.notna(row.get('Store Engagement', 0)) else 0,
                'foods': [], 'index': idx
            })

        for _, inv_row in df_inv.iterrows():
            store_name = inv_row['Store Name']
            store = next((s for s in stores if s['name'] == store_name), None)
            if not store: continue
            for i in range(1,4):
                name = inv_row.get(f'Food Name {i}')
                if pd.notna(name):
                    ext_food_img = ''
                    if pd.notna(inv_row.get(f'Food Image {i}')):
                        ext_food_img = inv_row.get(f'Food Image {i}','').replace('imgur.com/','i.imgur.com/') + '.jpg'
                    pid = store_to_pid.get(store_name)
                    food_img_url = ext_food_img
                    if parquet_df is not None and pid:
                        rowp = parquet_df.loc[pid]; slot_key = f"food{i}"; slot_choice = None
                        if _bytes_present(rowp.get(COL_MAP[slot_key][0])): slot_choice = slot_key
                        else: slot_choice = food_slot_map.get((store_name, str(name).strip()))
                        if slot_choice and _bytes_present(rowp.get(COL_MAP[slot_choice][0])):
                            food_img_url = url_for("serve_image", item_id=pid, col=slot_choice, w=360, h=640, mode="fit")
                        else:
                            if _bytes_present(rowp.get(COL_MAP["store"][0])):
                                food_img_url = url_for("serve_image", item_id=pid, col="store", w=360, h=640, mode="fit")

                    food = {
                        'name': name,
                        'quantity': inv_row.get(f'Food Quantity {i}', 0),
                        'price': inv_row.get(f'Food Price {i}', 0),
                        'image': food_img_url,
                        'engagement': 0,
                        'food_index': i,
                        'store': store_name,
                    }
                    eng_row = df_eng[df_eng['Store Name'] == store_name].iloc[0]
                    for j in range(1,4):
                        if eng_row.get(f'Food Name {j}') == name:
                            val = eng_row.get(f'Food Engagement {j}', 0)
                            food['engagement'] = val if pd.notna(val) else 0
                            break
                    store['foods'].append(food)
    return stores, df_eng, df_inv

def save_food_data(df_eng, df_inv):
    file_path = 'EXAMPLE RUN_Food Partners Database.xlsx'
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        df_eng.to_excel(writer, sheet_name='Engagement', index=False)
        df_inv.to_excel(writer, sheet_name='Inventory', index=False)

# -----------------------------
# NLP helpers
# -----------------------------
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(str(text).lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

def search_similar(query, df, column, threshold=0.4):
    df = df.copy()
    df['processed'] = df[column].apply(preprocess_text)
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(df['processed'])
    query_processed = preprocess_text(query)
    query_vector = vectorizer.transform([query_processed])
    sims = cosine_similarity(query_vector, vectors).flatten()
    indices = [i for i in range(len(sims)) if sims[i] > threshold]
    return [{'index': df.index[i], 'similarity': sims[i]} for i in indices]

# -----------------------------
# Jinja filter: IDR formatting
# -----------------------------
@app.template_filter('idr')
def format_idr(value):
    try:
        n = float(value)
    except Exception:
        return value
    return "IDR " + f"{n:,.0f}".replace(",", ".")

# -----------------------------
# Routes
# -----------------------------
# Login is now available at "/" AND "/Login" (backward compatible).
@app.route('/', methods=['GET', 'POST'])
@app.route('/Login', methods=['GET', 'POST'])
def login():
    global email_logged
    if request.method == 'POST':
        user_email = request.form['email']
        user_password = request.form['password']
        df_user, df_partner = get_db()
        user_match = df_user[(df_user['E-mail'] == user_email) & (df_user['Password'] == user_password)]
        partner_match = df_partner[(df_partner['E-mail'] == user_email) & (df_partner['Password'] == user_password)]
        if not user_match.empty or not partner_match.empty:
            email_logged = user_email
            session['email_logged'] = user_email
            # Go to EXPLORE after login
            return redirect(url_for('explore'))
        else:
            return render_template('login.htm', error="Invalid E-mail or Password. Please try again.")
    return render_template('login.htm')

@app.route('/dashboard')
def dashboard():
    logged_email = session.get('email_logged') or email_logged
    if logged_email:
        return f"<h1>Welcome to SAVR, {logged_email}!</h1><p>You are logged in.</p><a href='/logout'>Logout</a>"
    else:
        # redirect to the root login
        return redirect(url_for('login'))

# --------- EXPLORE now handles POST (add to cart) ----------
@app.route('/explore', methods=['GET', 'POST'])
def explore():
    logged_email = session.get('email_logged') or email_logged
    if not logged_email:
        return redirect(url_for('login'))

    if request.method == 'POST':
        store_name = request.form.get('store', '')
        food_name = request.form.get('food', '')
        stores, df_eng, df_inv = load_food_data()
        store = next((s for s in stores if s['name'] == store_name), None)
        if store:
            food = next((f for f in store['foods'] if f['name'] == food_name), None)
            if food and food.get('quantity', 0) > 0:
                inv_row = df_inv[df_inv['Store Name'] == store_name].index[0]
                for i in range(1, 4):
                    if df_inv.at[inv_row, f'Food Name {i}'] == food_name:
                        df_inv.at[inv_row, f'Food Quantity {i}'] = df_inv.at[inv_row, f'Food Quantity {i}'] - 1
                        break
                eng_row = df_eng[df_eng['Store Name'] == store_name].index[0]
                for j in range(1, 4):
                    if df_eng.at[eng_row, f'Food Name {j}'] == food_name:
                        cur = df_eng.at[eng_row, f'Food Engagement {j}'] or 0
                        df_eng.at[eng_row, f'Food Engagement {j}'] = cur + 1
                        break
                save_food_data(df_eng, df_inv)
                cart = session.get('cart', [])
                existing = next((it for it in cart if it['store'] == store_name and it['food'] == food_name), None)
                if existing:
                    existing['qty'] += 1
                else:
                    cart.append({'store': store_name, 'food': food_name, 'price': food.get('price', 0), 'qty': 1})
                session['cart'] = cart
        sel = request.args.get('store') or store_name
        return redirect(url_for('explore', store=sel))

    # GET
    stores, df_eng, df_inv = load_food_data()
    user_lat, user_lon = get_user_location(request.args.get('lat'), request.args.get('lon'))
    for store in stores:
        try:
            store['distance'] = haversine(user_lat, user_lon, float(store['lat']), float(store['lon']))
        except Exception:
            store['distance'] = 'N/A'

    search_restaurant = request.args.get('search_restaurant', '').strip()
    search_food = request.args.get('search_food', '').strip()
    selected_store_name = request.args.get('store')

    # Restaurant search
    filtered_stores = stores
    if search_restaurant:
        df_stores = pd.DataFrame([{'Store Name': s['name']} for s in stores])
        results = search_similar(search_restaurant, df_stores, 'Store Name', threshold=0.4)
        if results:
            matched_names = [df_stores.iloc[r['index']]['Store Name'] for r in results]
            filtered_stores = [s for s in stores if s['name'] in matched_names]
            if df_eng is not None and df_inv is not None:
                for name in matched_names:
                    idx = df_eng.index[df_eng['Store Name'] == name]
                    if len(idx):
                        cur_val = df_eng.loc[idx, 'Store Engagement'].fillna(0)
                        df_eng.loc[idx, 'Store Engagement'] = cur_val + 1
                save_food_data(df_eng, df_inv)
        else:
            q = search_restaurant.lower()
            substring_matches = [s for s in stores if q in s['name'].lower()]
            filtered_stores = substring_matches if substring_matches else stores

    # Food search
    all_foods = [f for s in filtered_stores for f in s['foods']]
    filtered_foods = all_foods
    if search_food and all_foods:
        df_foods = pd.DataFrame([{'Food Name': f['name']} for f in filtered_foods])
        results_food = search_similar(search_food, df_foods, 'Food Name', threshold=0.4)
        if results_food:
            filtered_foods = [filtered_foods[r['index']] for r in results_food]
        else:
            q = search_food.lower()
            substring = [f for f in filtered_foods if q in f['name'].lower()]
            filtered_foods = substring if substring else filtered_foods

    trending_stores = sorted(filtered_stores, key=lambda s: s['engagement'], reverse=True)[:5] if not selected_store_name else []
    trending_foods = sorted(filtered_foods, key=lambda f: f['engagement'], reverse=True)[:5]
    selected_store = next((s for s in stores if s['name'] == selected_store_name), None) if selected_store_name else None
    selected_foods = selected_store['foods'] if selected_store else []

    return render_template('0_explore.htm',
                           trending_stores=trending_stores,
                           trending_foods=trending_foods,
                           selected_store=selected_store,
                           selected_foods=selected_foods)

# -----------------------------
# CART: helpers and routes
# -----------------------------
def _build_catalog_index(stores):
    idx = {}
    for s in stores:
        for f in s.get('foods', []):
            idx[(s['name'], f['name'])] = {'image': f.get('image') or s.get('image') or '', 'price': f.get('price', 0)}
    return idx

@app.route('/cart', methods=['GET'])
def cart():
    items = session.get('cart', []) or []
    stores, _, _ = load_food_data()
    idx = _build_catalog_index(stores)

    view_items, grand_total = [], 0
    for i, item in enumerate(items):
        store = item.get('store'); food = item.get('food')
        qty = int(item.get('qty', 0)); price = float(item.get('price', 0))
        cat = idx.get((store, food), {})
        image = cat.get('image', item.get('image', ''))
        if 'price' in cat and cat['price']:
            price = float(cat['price'])
        line_total = price * qty; grand_total += line_total
        view_items.append({'index': i, 'store': store, 'food': food, 'qty': qty, 'price': price, 'line_total': line_total, 'image': image})

    return render_template('cart.htm', items=view_items, grand_total=grand_total)

@app.route('/cart/update', methods=['POST'])
def cart_update():
    action = request.form.get('action')
    try: idx = int(request.form.get('index','-1'))
    except Exception: idx = -1
    cart = session.get('cart', []) or []
    if 0 <= idx < len(cart):
        if action == 'inc':
            cart[idx]['qty'] = int(cart[idx].get('qty', 0)) + 1
        elif action == 'dec':
            new_qty = int(cart[idx].get('qty', 0)) - 1
            if new_qty <= 0: cart.pop(idx)
            else: cart[idx]['qty'] = new_qty
        elif action == 'remove':
            cart.pop(idx)
    session['cart'] = cart
    return redirect(url_for('cart'))

@app.route('/checkout/confirm', methods=['POST'])
def checkout_confirm():
    session['cart'] = []; flash("Payment confirmed (stub). Cart cleared.")
    return redirect(url_for('cart'))

# -----------------------------
# PROFILE
# -----------------------------
@app.route('/profile')
def profile():
    logged_email = session.get('email_logged') or email_logged
    if not logged_email:
        return redirect(url_for('login'))

    acct = find_account_by_email(logged_email)
    if not acct:
        acct = {"name": "Unknown", "email": logged_email, "role": "Unknown"}

    return render_template('profile.htm', profile=acct)

# -----------------------------
# Auth utilities
# -----------------------------
@app.route('/logout')
def logout():
    global email_logged
    email_logged = None
    session.pop('email_logged', None)
    # send back to root login
    return redirect(url_for('login'))

@app.route('/UserRegistration', methods=['GET', 'POST'])
def user_registration():
    global email_logged
    if request.method == 'POST':
        first_name = request.form['first_name']
        last_name = request.form.get('last_name', '')
        email = request.form['email']
        password = request.form['password']
        re_password = request.form['re_password']
        terms_read = request.form.get('terms_read')
        terms_agree = request.form.get('terms_agree')

        if not first_name or not email or not password or not re_password:
            return render_template('userregistration.htm', error="All fields except Last Name are required.")
        if password != re_password:
            return render_template('userregistration.htm', error="Passwords do not match.")
        if not terms_read or not terms_agree:
            return render_template('userregistration.htm', error="Please read and agree to the Terms & Conditions.")

        add_user_to_db(first_name, last_name, email, password)
        session['email_logged'] = email
        email_logged = email
        # go straight to explore
        return redirect(url_for('explore'))

    return render_template('userregistration.htm')

@app.route('/PartnerRegistration', methods=['GET', 'POST'])
def partner_registration():
    global email_logged
    if request.method == 'POST':
        partner_name = request.form['partner_name']
        owner_name = request.form['owner_name']
        nik_ktp = request.form['nik_ktp']
        email = request.form['email']
        password = request.form['password']
        re_password = request.form['re_password']
        terms_read = request.form.get('terms_read')
        terms_agree = request.form.get('terms_agree')

        if not partner_name or not owner_name or not nik_ktp or not email or not password or not re_password:
            return render_template('partnerregistration.htm', error="All fields are required.")
        if password != re_password:
            return render_template('partnerregistration.htm', error="Passwords do not match.")

        add_partner_to_db(partner_name, owner_name, nik_ktp, email, password)
        session['email_logged'] = email
        email_logged = email
        # go straight to explore
        return redirect(url_for('explore'))

    return render_template('partnerregistration.htm')

@app.route('/TermsAndConditions')
def terms_and_conditions():
    return "<h1>Terms & Conditions</h1><p>Details of Terms & Conditions go here.</p>"

if __name__ == '__main__':
    app.run(debug=True)
