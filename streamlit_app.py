import streamlit as st
import pandas as pd
import numpy as np
import io
import requests
from urllib.parse import urlparse, quote

def weserv_proxy(url: str) -> str:
    try:
        # Proxy through images.weserv.nl (public image proxy). If it fails, caller will fallback.
        # We avoid double-encoding the scheme.
        safe = quote(url, safe=":/%?#[]@!$&'()*+,;=")
        return f"https://images.weserv.nl/?url={safe}"
    except Exception:
        return url

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


@st.cache_data(show_spinner=False)
def fetch_image_bytes(url: str):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123 Safari/537.36",
            "Referer": "https://www.linkedin.com/",
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        }
        r = requests.get(url, headers=headers, timeout=6)
        if r.status_code == 200 and r.headers.get("Content-Type", "").startswith("image"):
            # Limit size to ~5MB
            content = r.content[:5*1024*1024]
            return content
    except Exception:
        return None
    return None
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

# -------- Fixed column mapping (per user)
col_first = "firstName" if "firstName" in df_use.columns else None
col_last = "lastName" if "lastName" in df_use.columns else None
col_company = "companyName" if "companyName" in df_use.columns else None
col_job = "linkedinHeadline" if "linkedinHeadline" in df_use.columns else None
col_photo = "linkedinprofileImageurl" if "linkedinprofileImageurl" in df_use.columns else None
col_location = "linkedinJobLocation" if "linkedinJobLocation" in df_use.columns else None

# -------- KPIs
st.markdown("### 📈 Statistiques")
colK1, colK2 = st.columns(2)
colK1.metric("Leads (total)", f"{len(df_use):,}")
if col_company:
    colK2.metric("Entreprises uniques", f"{df_use[col_company].nunique():,}")

# -------- Top filters (3 columns)
st.markdown("### 🎛️ Filtres principaux")
top_filters = {}

followers_col = None
connections_col = None
company_size_col = None
company_founded_col = None

for c in df_use.columns:
    lc = c.lower()
    if followers_col is None and "followers" in lc:
        followers_col = c
    if connections_col is None and ("connection" in lc or "connexion" in lc):
        connections_col = c
    if company_size_col is None and "companysize" in lc:
        company_size_col = c
    if company_founded_col is None and "founded" in lc:
        company_founded_col = c

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

cols = st.columns(3, gap="large")
filter_specs = [
    ("Nombre de followers", followers_col, "num"),
    ("Nombre de connexions", connections_col, "num"),
    ("Taille de l'entreprise", company_size_col, "cat"),
    ("Création de l'entreprise", company_founded_col, "num"),
]
for i, (label, c, kind) in enumerate(filter_specs):
    if c and c in df_use.columns:
        with cols[i % 3]:
            if kind == "num":
                add_numeric_slider(label, df_use[c], c)
            else:
                add_categorical_multiselect(label, df_use[c], c, max_unique=50)

# -------- Sidebar: only text search filters (unique keys)
st.sidebar.header("🔎 Recherche (texte)")
text_filters = {}
skip_cols = set([x for x in [followers_col, connections_col, company_size_col, company_founded_col] if x])
for c in df_use.columns:
    if c in skip_cols:
        continue
    if pd.api.types.is_numeric_dtype(df_use[c]) or pd.api.types.is_datetime64_any_dtype(df_use[c]):
        continue
    if c == col_photo:
        continue
    val = st.sidebar.text_input(prettify_label(c), "", key=f"txt_{c}")
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
            mask &= (rng_mask | s_num.isna())
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
page_size = st.sidebar.selectbox("Taille de page", [12, 24, 48, 96], index=1, key="page_size")
total_rows = len(filtered)
total_pages = (total_rows - 1) // page_size + 1 if total_rows > 0 else 1
page = st.sidebar.number_input("Page", min_value=1, max_value=max(1, total_pages), value=1, step=1, key="page_num")
start, end = (page - 1) * page_size, (page - 1) * page_size + page_size

st.markdown(f"**{total_rows:,} leads** après filtres • Page **{page}/{total_pages}**")

# -------- Cards
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
    if col_first and pd.notna(row.get(col_first, np.nan)):
        parts.append(str(row[col_first]).strip())
    if col_last and pd.notna(row.get(col_last, np.nan)):
        parts.append(str(row[col_last]).strip())
    return " ".join(parts) if parts else "(Sans nom)"

def get_company_job(row):
    comp = str(row[col_company]).strip() if col_company and pd.notna(row.get(col_company, np.nan)) else ""
    job = str(row[col_job]).strip() if col_job and pd.notna(row.get(col_job, np.nan)) else ""
    return f"{comp} — {job}" if comp and job else (comp or job or "")



def render_card(rec):
    fetched = False
    if col_photo:
        img = rec.get(col_photo)
        if isinstance(img, str) and looks_like_url(img):
            data = fetch_image_bytes(img)
            if data:
                st.image(io.BytesIO(data), use_container_width=True)
                fetched = True
            else:
                # Try weserv proxy url
                proxy_url = weserv_proxy(img)
                try:
                    st.image(proxy_url, use_container_width=True)
                    fetched = True
                except Exception:
                    fetched = False
    if not fetched:
        st.markdown('<div style="width:100%;aspect-ratio:1.8;background:rgba(0,0,0,0.03);border-radius:10px;"></div>', unsafe_allow_html=True)

    name = get_display_name(rec)
    sub = get_company_job(rec)
    loc = f'<div class="lead-loc">{rec[col_location]}</div>' if col_location and pd.notna(rec.get(col_location)) else ""
    html = f'''
        <div class="lead-card">
            <div class="lead-name">{name}</div>
            <div class="lead-sub">{sub}</div>
            {loc}
        </div>
    '''
    st.markdown(html, unsafe_allow_html=True)



rows = []
recs = list(subset.to_dict(orient="records"))
for i in range(0, len(recs), cards_per_row):
    rows.append(recs[i:i+cards_per_row])

for row in rows:
    cols = st.columns(cards_per_row, gap="large")
    for col, rec in zip(cols, row):
        with col:
            render_card(rec)


with st.expander("🔧 Debug images"):
    sample_urls = []
    if col_photo and col_photo in df_use.columns:
        sample_urls = [u for u in df_use[col_photo].dropna().astype(str).head(5).tolist() if looks_like_url(u)]
    if not sample_urls:
        st.write("Aucune URL valide détectée.")
    else:
        for u in sample_urls:
            st.write("URL:", u)
            data = fetch_image_bytes(u)
            st.write("fetch_image_bytes:", "OK" if data else "None")
            if not data:
                st.write("weserv proxy test:")
                st.image(weserv_proxy(u))
\n\nst.download_button("⬇️ Télécharger le CSV filtré", data=filtered.to_csv(index=False).encode("utf-8"),
                   file_name="leads_filtres.csv", mime="text/csv")

with st.expander("📊 Statistiques détaillées"):
    try:
        st.write(filtered.describe(include="all", datetime_is_numeric=True))
    except TypeError:
        st.write(filtered.describe(include="all"))
