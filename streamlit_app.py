
import streamlit as st
import pandas as pd
import numpy as np
from urllib.parse import urlparse

st.set_page_config(page_title="Lead Manager", page_icon="📇", layout="wide")

# ---------- Helpers
def prettify_label(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = s.replace("_", " ").replace("-", " ")
    for k in ["linkedin", "LinkedIn", "Linkedin"]:
        s = s.replace(k, "")
    s = " ".join(s.split())
    return s.strip().title()

@st.cache_data(show_spinner=False)
def read_csv_any(uploaded) -> pd.DataFrame:
    try:
        return pd.read_csv(uploaded, sep=None, engine="python", low_memory=False)
    except Exception:
        try:
            uploaded.seek(0)
        except Exception:
            pass
        return pd.read_csv(uploaded, low_memory=False)

def coerce_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        try:
            if df[col].dtype == object:
                sample = df[col].dropna().head(50).astype(str)
                if sample.str.contains(r"\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{2,4}", regex=True).mean() > 0.5:
                    df[col] = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True, utc=False)
            if pd.api.types.is_datetime64tz_dtype(df[col]):
                df[col] = df[col].dt.tz_localize(None)
        except Exception:
            pass
    return df

def looks_like_url(s: str) -> bool:
    if not isinstance(s, str) or not s:
        return False
    if s.startswith("urn:"):
        return False
    try:
        u = urlparse(s)
        return u.scheme in ("http", "https") and bool(u.netloc)
    except Exception:
        return False

# Columns to ignore in filters
IGNORE_FILTERS = [
    "error", "company slug", "job date range", "profile image urn", "profile urn",
    "school urn", "school company slug", "mutual connections", "profile url",
    "scraper fullname", "school date range"
]

st.title("📇 Lead Manager")
st.caption("Grille de cartes + filtres optimisés")

uploaded = st.file_uploader("Déposez votre CSV ici", type=["csv"], accept_multiple_files=False)
if uploaded is None:
    st.info("Aucun fichier importé. Déposez un CSV pour commencer.")
    st.stop()

df = read_csv_any(uploaded)
df = coerce_datetimes(df)

# Filter out ignored columns
def is_ignored(col: str) -> bool:
    lc = col.lower()
    return any(x in lc for x in IGNORE_FILTERS)
usable_cols = [c for c in df.columns if not is_ignored(c)]
df_use = df[usable_cols].copy()

# -------- KPIs
st.markdown("### 📈 Statistiques")
colK1, colK2 = st.columns(2)
colK1.metric("Leads (total)", f"{len(df_use):,}")

# -------- Mapping de colonnes
st.sidebar.header("🧩 Mapping des colonnes")
def pick(label, default=None):
    opts = ["—"] + list(df_use.columns)
    idx = 0
    if default in df_use.columns:
        try:
            idx = opts.index(default)
        except ValueError:
            idx = 0
    return st.sidebar.selectbox(label, opts, index=idx)

# Defaults per your mapping
default_photo = "linkedinprofileImageurl" if "linkedinprofileImageurl" in df_use.columns else None
default_company = "companyName" if "companyName" in df_use.columns else None

col_first = pick("First Name", "firstName" if "firstName" in df_use.columns else None)
col_last = pick("Last Name", "lastName" if "lastName" in df_use.columns else None)
col_full = pick("Full Name (optionnel)", None)  # a supprimer -> None
col_company = pick("Company", default_company)
col_job = pick("Job / Title", "linkedinHeadline" if "linkedinHeadline" in df_use.columns else None)
col_photo = pick("Photo URL", default_photo)
col_location = pick("Location", "linkedinJobLocation" if "linkedinJobLocation" in df_use.columns else None)

if col_company != "—":
    colK2.metric("Entreprises uniques", f"{df_use[col_company].nunique():,}")

# -------- Top filters (custom labels + remove employees count)
st.markdown("
### 🎛️ Filtres principaux
top_filters = {}

# Identify columns for filters (labels FR)
followers_col = None
connections_col = None
company_size_col = None
company_founded_col = None

for c in df_use.columns:
    lc = c.lower()
    if followers_col is None and ("followerscount" in lc or lc.endswith("followers") or "followers count" in lc):
        followers_col = c
    if connections_col is None and ("connection" in lc or "connexion" in lc):
        connections_col = c
    if company_size_col is None and ("companysize" in lc or "company size" in lc or lc == "size"):
        company_size_col = c
    if company_founded_col is None and ("companyfounded" in lc or "founded" in lc):
        company_founded_col = c

# Widgets helpers
def add_numeric_slider(label_fr, series, key_name):
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().any():
        lo, hi = float(np.nanmin(s)), float(np.nanmax(s))
        if lo != hi:
            v = st.slider(label_fr, lo, hi, (lo, hi), key=f"sl_{key_name}")
            top_filters[key_name] = ("num_range_nanpass", v)

def add_categorical_multiselect(label_fr, series, key_name, max_unique=50):
    uniques = series.dropna().astype(str).unique()
    if 1 <= len(uniques) <= max_unique:
        v = st.multiselect(label_fr, sorted(map(str, uniques)), key=f"ms_{key_name}")
        if v:
            top_filters[key_name] = ("in", set(v))

# Build the list of available filters (label, type, colname)
filter_specs = []
if followers_col and followers_col in df_use:
    filter_specs.append(("Nombre de followers", "num", followers_col))
if connections_col and connections_col in df_use:
    filter_specs.append(("Nombre de connexions", "num", connections_col))
if company_size_col and company_size_col in df_use:
    filter_specs.append(("Taille de l'entreprise", "cat", company_size_col))
if company_founded_col and company_founded_col in df_use:
    filter_specs.append(("Création de l'entreprise", "num", company_founded_col))

cols = st.columns(3, gap="large")
for idx, (label_fr, kind, colname) in enumerate(filter_specs):
    with cols[idx % 3]:
        if kind == "num":
            add_numeric_slider(label_fr, df_use[colname], colname)
        else:
            add_categorical_multiselect(label_fr, df_use[colname], colname, max_unique=50)

🔎 Recherche (texte)")
text_filters = {}
skip_cols = set([c for c in [followers_col, connections_col, company_size_col, company_founded_col] if c])
candidates_text = []
for c in df_use.columns:
    if c in skip_cols:
        continue
    if pd.api.types.is_numeric_dtype(df_use[c]) or pd.api.types.is_datetime64_any_dtype(df_use[c]):
        continue
    if c == col_photo or c == "—":
        continue
    candidates_text.append(c)

with st.sidebar.expander("Choisir les colonnes à rechercher", expanded=False):
    default_pretty = [prettify_label(c) for c in candidates_text[:6]]
    selected_text_cols_pretty = st.multiselect("Colonnes texte", options=[prettify_label(c) for c in candidates_text], default=default_pretty)
pretty_to_real = {prettify_label(c): c for c in candidates_text}
selected_real = [pretty_to_real[p] for p in selected_text_cols_pretty if p in pretty_to_real]
for c in selected_real:
    val = st.sidebar.text_input(prettify_label(c), "")
    if val:
        text_filters[c] = ("contains", val.lower())

# -------- Apply filters
def apply_all_filters(df_in: pd.DataFrame):
    df_out = df_in.copy()
    mask = pd.Series([True] * len(df_out), index=df_out.index)
    for col_key, (ftype, val) in top_filters.items():
        if col_key not in df_out:
            continue
        s = df_out[col_key]
        if ftype == "num_range_nanpass":
            s_num = pd.to_numeric(s, errors="coerce")
            lo, hi = val
            rng_mask = s_num.between(lo, hi)
            mask &= (rng_mask | s_num.isna())  # NaN passes
        elif ftype == "in":
            mask &= s.astype(str).isin(val)
    for col_key, (ftype, val) in text_filters.items():
        if col_key not in df_out:
            continue
        s = df_out[col_key]
        mask &= s.astype(str).str.lower().str.contains(val, na=False)
    return df_out[mask]

filtered = apply_all_filters(df_use)

# -------- Pagination
st.sidebar.header("🧭 Pagination")
page_size = st.sidebar.selectbox("Taille de page", [12, 24, 48, 96], index=1)
total_rows = len(filtered)
total_pages = (total_rows - 1) // page_size + 1 if total_rows > 0 else 1
page = st.sidebar.number_input("Page", min_value=1, max_value=max(1, total_pages), value=1, step=1)
start, end = (page - 1) * page_size, (page - 1) * page_size + page_size

# Header count reflects filtered rows
st.markdown(f"**{total_rows:,} leads** après filtres • Page **{page}/{total_pages}**")

# -------- Card Grid with CSS borders
st.markdown("### 🧾 Résultats")

st.markdown(
    """
    <style>
    .lead-card {
        border: 1px solid rgba(0,0,0,0.1);
        border-radius: 12px;
        padding: 12px;
        height: 100%;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .lead-card img {
        border-radius: 10px;
        width: 100%;
        height: auto;
        object-fit: cover;
    }
    .lead-name { font-weight: 700; margin-top: 6px; }
    .lead-sub { color: rgba(49,51,63,0.7); font-size: 0.95rem; }
    .lead-loc { color: rgba(49,51,63,0.6); font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

subset = filtered.iloc[start:end]
cards_per_row = st.selectbox("Cartes par ligne", [3, 4, 6], index=1, key="cards_per_row_selector")

def get_display_name(row):
    parts = []
    if col_first != "—" and pd.notna(row.get(col_first, np.nan)) and str(row.get(col_first)).strip():
        parts.append(str(row[col_first]).strip())
    if col_last != "—" and pd.notna(row.get(col_last, np.nan)) and str(row.get(col_last)).strip():
        parts.append(str(row[col_last]).strip())
    if parts:
        return " ".join(parts)
    if col_full and col_full != "—" and pd.notna(row.get(col_full, np.nan)) and str(row.get(col_full)).strip():
        return str(row[col_full]).strip()
    return "(Sans nom)"

def get_company_job(row):
    comp = str(row[col_company]).strip() if col_company != "—" and pd.notna(row.get(col_company, np.nan)) else ""
    job = str(row[col_job]).strip() if col_job != "—" and pd.notna(row.get(col_job, np.nan)) else ""
    if comp and job:
        return f"{comp} — {job}"
    return comp or job or ""

def render_card(rec):
    # Image
    img_html = ""
    if col_photo != "—":
        img = rec.get(col_photo)
        if isinstance(img, str) and looks_like_url(img):
            img_html = f'<img src="{img}" alt="photo">'
    if not img_html:
        # Placeholder
        img_html = '<div style="width:100%;aspect-ratio:1.8;background:rgba(0,0,0,0.03);border-radius:10px;"></div>'
    name = get_display_name(rec)
    sub = get_company_job(rec)
    loc = ""
    if col_location != "—" and pd.notna(rec.get(col_location, np.nan)):
        loc = f'<div class="lead-loc">{str(rec[col_location])}</div>'
    html = f'''
        <div class="lead-card">
            {img_html}
            <div class="lead-name">{name}</div>
            <div class="lead-sub">{sub}</div>
            {loc}
        </div>
    '''
    st.markdown(html, unsafe_allow_html=True)

# Build grid
rows = []
recs = list(subset.to_dict(orient="records"))
for i in range(0, len(recs), cards_per_row):
    rows.append(recs[i:i+cards_per_row])

for row in rows:
    cols = st.columns(cards_per_row, gap="large")
    for col, rec in zip(cols, row):
        with col:
            render_card(rec)

# -------- Export & Stats table
st.download_button("⬇️ Télécharger le CSV filtré", data=filtered.to_csv(index=False).encode("utf-8"),
                   file_name="leads_filtres.csv", mime="text/csv")

with st.expander("📊 Statistiques détaillées"):
    try:
        st.write(filtered.describe(include="all", datetime_is_numeric=True))
    except TypeError:
        st.write(filtered.describe(include="all"))
