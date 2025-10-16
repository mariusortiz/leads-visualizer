
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
    if s.startswith("urn:"):  # LinkedIn URN or similar
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
colK1, colK2, colK3, colK4 = st.columns(4)
colK1.metric("Leads (total)", f"{len(df_use):,}")
# Entreprises uniques si colonne sélectionnée plus bas

# -------- Mapping de colonnes (utilisateur)
st.sidebar.header("🧩 Mapping des colonnes")
def pick(label, default=None):
    opts = ["—"] + list(df_use.columns)
    idx = 0
    if default in df_use.columns:
        idx = opts.index(default) if default in opts else 0
    return st.sidebar.selectbox(label, opts, index=idx)

# Tentatives de défauts par heuristique
def guess(cands):
    for c in df_use.columns:
        lc = c.lower()
        for cand in cands:
            if cand in lc:
                return c
    return None

col_first = pick("First Name", guess(["first name","firstname","first_name","given name"]))
col_last = pick("Last Name", guess(["last name","lastname","last_name","family name","surname"]))
col_full = pick("Full Name (optionnel)", guess(["full name","fullname","name"]))
col_company = pick("Company", guess(["company","company name","employer","organization","org name"]))
col_job = pick("Job / Title", guess(["job","job title","title","position","role","headline"]))
col_photo = pick("Photo URL", guess(["profile image url","profile picture","image url","photo url","picture url","avatar","img"]))
col_location = pick("Location", guess(["location","city","country","region"]))

# Update KPIs with company uniques if mapped
if col_company != "—":
    colK2.metric("Entreprises uniques", f"{df_use[col_company].nunique():,}")
# Followers / Employees moy. si présentes
followers_col = guess(["followers count","followers","follower count"])
employees_col = guess(["employee count","employees","employee counts","company employees"])
if followers_col:
    s = pd.to_numeric(df_use[followers_col], errors="coerce")
    if s.notna().any():
        colK3.metric("Followers (moy.)", f"{float(s.mean()):,.0f}")
if employees_col:
    s = pd.to_numeric(df_use[employees_col], errors="coerce")
    if s.notna().any():
        colK4.metric("Employés (moy.)", f"{float(s.mean()):,.0f}")

# -------- Top filters (num/cat spécifiques)
st.markdown("### 🎛️ Filtres principaux")
top_filters = {}
def add_numeric_slider(label, series, key_name):
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().any():
        lo, hi = float(np.nanmin(s)), float(np.nanmax(s))
        if lo != hi:
            v = st.slider(prettify_label(label), lo, hi, (lo, hi))
            top_filters[key_name] = ("num_range", v)

def add_categorical_multiselect(label, series, key_name, max_unique=50):
    uniques = series.dropna().astype(str).unique()
    if 1 <= len(uniques) <= max_unique:
        v = st.multiselect(prettify_label(label), sorted(map(str, uniques)))
        if v:
            top_filters[key_name] = ("in", set(v))

# Let user choose how many cards per row
cards_per_row = st.selectbox("Cartes par ligne", [3, 4, 6], index=1)

# Specific filters per your request
company_followers_col = guess(["company followers count","company followers","org followers"])
followers_col = followers_col
connections_col = guess(["connection count","connections","connexion count"])
employees_col = employees_col
company_size_col = guess(["company size","size","organization size"])
company_founded_col = guess(["company founded","founded","founded year","foundation year","year founded"])

cols_top = st.columns(3)
with cols_top[0]:
    if company_followers_col and company_followers_col in df_use:
        add_numeric_slider(company_followers_col, df_use[company_followers_col], company_followers_col)
    if followers_col and followers_col in df_use:
        add_numeric_slider(followers_col, df_use[followers_col], followers_col)
with cols_top[1]:
    if connections_col and connections_col in df_use:
        add_numeric_slider(connections_col, df_use[connections_col], connections_col)
    if employees_col and employees_col in df_use:
        add_numeric_slider(employees_col, df_use[employees_col], employees_col)
with cols_top[2]:
    if company_size_col and company_size_col in df_use:
        add_categorical_multiselect(company_size_col, df_use[company_size_col], company_size_col, max_unique=50)
    if company_founded_col and company_founded_col in df_use:
        add_numeric_slider(company_founded_col, df_use[company_founded_col], company_founded_col)

# -------- Sidebar: only text search filters
st.sidebar.header("🔎 Recherche (texte)")
text_filters = {}
# candidate text columns (excluding numeric/datetime and already filtered ones)
candidates_text = []
skip_cols = set([c for c in [company_followers_col, followers_col, connections_col, employees_col, company_size_col, company_founded_col] if c])
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
        if ftype == "num_range":
            s_num = pd.to_numeric(s, errors="coerce")
            lo, hi = val
            mask &= s_num.between(lo, hi)
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
st.markdown(f"**{total_rows:,} leads** après filtres • Page **{page}/{total_pages}**")

# -------- Grid Card View
st.markdown("### 🧾 Résultats")
subset = filtered.iloc[start:end]

def get_display_name(row):
    # Priority: First + Last, else Full Name, else fallback
    parts = []
    if col_first != "—" and pd.notna(row.get(col_first, np.nan)) and str(row.get(col_first)).strip():
        parts.append(str(row[col_first]).strip())
    if col_last != "—" and pd.notna(row.get(col_last, np.nan)) and str(row.get(col_last)).strip():
        parts.append(str(row[col_last]).strip())
    if parts:
        return " ".join(parts)
    if col_full != "—" and pd.notna(row.get(col_full, np.nan)) and str(row.get(col_full)).strip():
        return str(row[col_full]).strip()
    return "(Sans nom)"

def get_company_job(row):
    comp = str(row[col_company]).strip() if col_company != "—" and pd.notna(row.get(col_company, np.nan)) else ""
    job = str(row[col_job]).strip() if col_job != "—" and pd.notna(row.get(col_job, np.nan)) else ""
    if comp and job:
        return f"{comp} — {job}"
    return comp or job or ""

# Build rows of columns
rows = []
cards = list(subset.to_dict(orient="records"))
for i in range(0, len(cards), cards_per_row):
    rows.append(cards[i:i+cards_per_row])

for row in rows:
    cols = st.columns(cards_per_row, gap="large")
    for col, rec in zip(cols, row):
        with col:
            # Image on top (if valid URL)
            if col_photo != "—":
                img = rec.get(col_photo)
                if isinstance(img, str) and looks_like_url(img):
                    try:
                        st.image(img, use_container_width=True)
                    except Exception:
                        st.write("")
                else:
                    st.write("")
            # Title: Lead Name
            st.markdown(f"**{get_display_name(rec)}**")
            # Subtitle: Company — Job (no dates)
            cj = get_company_job(rec)
            if cj:
                st.caption(cj)
            # Location (optional)
            if col_location != "—" and pd.notna(rec.get(col_location, np.nan)):
                st.caption(str(rec[col_location]))

# -------- Export & Stats table
st.download_button("⬇️ Télécharger le CSV filtré", data=filtered.to_csv(index=False).encode("utf-8"),
                   file_name="leads_filtres.csv", mime="text/csv")

with st.expander("📊 Statistiques détaillées"):
    try:
        st.write(filtered.describe(include="all", datetime_is_numeric=True))
    except TypeError:
        st.write(filtered.describe(include="all"))
