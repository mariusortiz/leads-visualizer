
import streamlit as st
import pandas as pd
import numpy as np
import io
import requests
from urllib.parse import urlparse, quote

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
        uploaded.seek(0)
        return pd.read_csv(uploaded, low_memory=False)

@st.cache_data(show_spinner=False)
def fetch_image_bytes(url: str):
    """Télécharge une image côté serveur avec des entêtes proches d’un vrai navigateur."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123 Safari/537.36",
            "Referer": "https://www.linkedin.com/",
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        }
        r = requests.get(url, headers=headers, timeout=6)
        if r.status_code == 200 and r.headers.get("Content-Type", "").startswith("image"):
            return r.content
    except Exception:
        return None
    return None

def weserv_proxy(url: str) -> str:
    """Proxy via images.weserv.nl pour contourner certains blocages."""
    safe = quote(url, safe=":/%?#[]@!$&'()*+,;=")
    return f"https://images.weserv.nl/?url={safe}"

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

# ---------- UI
st.title("📇 Lead Manager")
st.caption("Affichage cartes + filtres (haut et barre latérale)")

uploaded = st.file_uploader("Déposez votre fichier CSV", type=["csv"])
if uploaded is None:
    st.info("Déposez un fichier CSV pour commencer.")
    st.stop()

df = read_csv_any(uploaded)

# --- Colonnes fixes selon ton fichier
col_first = "firstName" if "firstName" in df.columns else None
col_last = "lastName" if "lastName" in df.columns else None
col_company = "companyName" if "companyName" in df.columns else None
col_job = "linkedinHeadline" if "linkedinHeadline" in df.columns else None
col_photo = "linkedinProfileImageUrl" if "linkedinProfileImageUrl" in df.columns else None
col_location = "linkedinJobLocation" if "linkedinJobLocation" in df.columns else None

# --- Statistiques
st.markdown("### 📈 Statistiques")
c1, c2 = st.columns(2)
c1.metric("Leads (total)", f"{len(df):,}")
if col_company:
    c2.metric("Entreprises uniques", f"{df[col_company].nunique():,}")

# --- Détection des colonnes pour filtres principaux
followers_col = None
connections_col = None
company_size_col = None
company_founded_col = None

for c in df.columns:
    lc = c.lower()
    if followers_col is None and ("followers" in lc or "followerscount" in lc):
        followers_col = c
    if connections_col is None and ("connection" in lc or "connexion" in lc):
        connections_col = c
    if company_size_col is None and ("companysize" in lc or "company size" in lc or lc == "size"):
        company_size_col = c
    if company_founded_col is None and ("companyfounded" in lc or "founded" in lc or "foundation year" in lc):
        company_founded_col = c

# --- Filtres principaux (3 colonnes)
st.markdown("### 🎛️ Filtres principaux")
top_filters = {}

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

cols_top = st.columns(3, gap="large")
filter_specs = []
if followers_col:        filter_specs.append(("Nombre de followers", followers_col, "num"))
if connections_col:      filter_specs.append(("Nombre de connexions", connections_col, "num"))
if company_size_col:     filter_specs.append(("Taille de l'entreprise", company_size_col, "cat"))
if company_founded_col:  filter_specs.append(("Création de l'entreprise", company_founded_col, "num"))

for i, (label, colname, kind) in enumerate(filter_specs):
    with cols_top[i % 3]:
        if kind == "num":
            add_numeric_slider(label, df[colname], colname)
        else:
            add_categorical_multiselect(label, df[colname], colname, max_unique=50)

# --- Filtres texte en sidebar (keys uniques)
st.sidebar.header("🔎 Recherche (texte)")
text_filters = {}
skip_cols = {x for x in [followers_col, connections_col, company_size_col, company_founded_col] if x}
for c in df.columns:
    if c in skip_cols:
        continue
    if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_datetime64_any_dtype(df[c]):
        continue
    if c == col_photo:
        continue
    val = st.sidebar.text_input(prettify_label(c), "", key=f"txt_{c}")
    if val:
        text_filters[c] = ("contains", val.lower())

# --- Application des filtres
def apply_all_filters(df_in: pd.DataFrame):
    df_out = df_in.copy()
    mask = pd.Series([True] * len(df_out), index=df_out.index)
    # top filters
    for col_key, (ftype, val) in top_filters.items():
        if col_key not in df_out:
            continue
        s = df_out[col_key]
        if ftype == "num_range_nanpass":
            s_num = pd.to_numeric(s, errors="coerce")
            lo, hi = val
            rng = s_num.between(lo, hi)
            mask &= (rng | s_num.isna())  # NaN passent
        elif ftype == "in":
            mask &= s.astype(str).isin(val)
    # text filters
    for col_key, (ftype, val) in text_filters.items():
        if col_key not in df_out:
            continue
        s = df_out[col_key]
        mask &= s.astype(str).str.lower().str.contains(val, na=False)
    return df_out[mask]

filtered = apply_all_filters(df)

# --- Pagination
st.sidebar.header("🧭 Pagination")
page_size = st.sidebar.selectbox("Taille de page", [12, 24, 48, 96], index=1, key="page_size")
total_rows = len(filtered)
total_pages = (total_rows - 1) // page_size + 1 if total_rows > 0 else 1
page = st.sidebar.number_input("Page", min_value=1, max_value=max(1, total_pages), value=1, step=1, key="page_num")
start, end = (page - 1) * page_size, (page - 1) * page_size + page_size

st.markdown(f"**{total_rows:,} leads** après filtres • Page **{page}/{total_pages}**")


# --- Résultats (cards)
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
    .lead-name { font-weight: 700; margin-top: 6px; }
    .lead-sub { color: rgba(49,51,63,0.7); font-size: 0.95rem; }
    .lead-loc { color: rgba(49,51,63,0.6); font-size: 0.85rem; }
    </style>
    """, unsafe_allow_html=True
)

cards_per_row = st.selectbox("Cartes par ligne", [3, 4, 6], index=1, key="cards_per_row_selector")

def get_display_name(row):
    parts = []
    if col_first and pd.notna(row.get(col_first, np.nan)):
        parts.append(str(row[col_first]).strip())
    if col_last and pd.notna(row.get(col_last, np.nan)):
        parts.append(str(row[col_last]).strip())
    return " ".join(parts) if parts else "(Sans nom)"

def render_card(rec):
    st.markdown('<div class="lead-card">', unsafe_allow_html=True)

    # Image
    if col_photo and pd.notna(rec.get(col_photo, None)) and looks_like_url(str(rec[col_photo])):
        url = str(rec[col_photo])
        data = fetch_image_bytes(url)
        if data:
            st.image(io.BytesIO(data), use_container_width=True)
        else:
            st.image(weserv_proxy(url), use_container_width=True)
    else:
        st.markdown('<div style="width:100%;aspect-ratio:1.8;background:#f0f0f0;border-radius:10px;"></div>', unsafe_allow_html=True)

    # Texte
    name = get_display_name(rec)
    company = str(rec[col_company]) if col_company and pd.notna(rec.get(col_company)) else ""
    job = str(rec[col_job]) if col_job and pd.notna(rec.get(col_job)) else ""
    location = str(rec[col_location]) if col_location and pd.notna(rec.get(col_location)) else ""

    st.markdown(f"**{name}**")
    st.caption(f"{company} — {job}" if company or job else "")
    st.caption(location)
    st.markdown("</div>", unsafe_allow_html=True)

# Grille
subset = filtered.iloc[start:end]
rows = []
recs = list(subset.to_dict(orient="records"))
for i in range(0, len(recs), cards_per_row):
    rows.append(recs[i:i + cards_per_row])

for row in rows:
    cols = st.columns(cards_per_row)
    for col, rec in zip(cols, row):
        with col:
            render_card(rec)

# Export
st.download_button(
    "⬇️ Télécharger le CSV filtré",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="leads_filtres.csv",
    mime="text/csv",
)
