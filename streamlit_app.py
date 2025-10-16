
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Lead Manager", page_icon="📇", layout="wide")

# -------- Utils
def prettify_label(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = s.replace("_", " ").replace("-", " ")
    s = s.replace("linkedin", "").replace("Linkedin", "").replace("LinkedIn", "")
    s = " ".join(s.split())  # collapse spaces
    return s.strip().title()

def find_col(df: pd.DataFrame, candidates):
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        lc = cand.lower()
        # exact
        if lc in cols:
            return cols[lc]
        # contains
        for k in cols:
            if lc in k:
                return cols[k]
    return None

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

# Columns to completely ignore in filters/display (case-insensitive contains match)
IGNORE_FILTERS = [
    "error", "company slug", "job date range", "profile image urn", "profile urn",
    "school urn", "school company slug", "mutual connections", "profile url",
    "scraper fullname", "school date range"
]

# Semantic candidates for special fields
CAND = {
    "first_name": ["first name", "firstname", "first_name", "given name"],
    "last_name": ["last name", "lastname", "last_name", "family name", "surname"],
    "full_name": ["full name", "name", "profile name", "contact name"],
    "company": ["company", "company name", "employer", "organization", "org name"],
    "job": ["job", "job title", "title", "position", "role", "headline"],
    "photo": ["profile image url", "profile picture url", "image url", "photo url", "picture url", "avatar"],
    "followers": ["followers count", "followers", "follower count"],
    "company_followers": ["company followers count", "company followers", "org followers"],
    "connections": ["connection count", "connections", "connexion count"],
    "employees": ["employee count", "employees", "employee counts", "company employees"],
    "company_size": ["company size", "size", "organization size"],
    "company_founded": ["company founded", "founded", "founded year", "foundation year", "year founded"],
    "location": ["location", "city", "country", "region"]
}

st.title("📇 Lead Manager")
st.caption("Filtres puissants + affichage en cartes + statistiques rapides")

uploaded = st.file_uploader("Déposez votre CSV ici", type=["csv"], accept_multiple_files=False)
if uploaded is None:
    st.info("Aucun fichier importé. Déposez un CSV pour commencer.")
    st.stop()

df = read_csv_any(uploaded)
df = coerce_datetimes(df)

# Hide ignored columns entirely from display & filters (non-destructive: keep original for cards)
def is_ignored(col: str) -> bool:
    c = col.lower()
    return any(x in c for x in IGNORE_FILTERS)

usable_cols = [c for c in df.columns if not is_ignored(c)]
df_use = df[usable_cols].copy()

# Detect key columns
key = {k: find_col(df_use, v) for k, v in CAND.items()}

# ---------------- KPIs (top of page)
# Compute a few robust stats
unique_companies = df_use[key["company"]].nunique() if key["company"] else np.nan
total_leads = len(df_use)
# Use safe means
def safe_mean(colname):
    if not colname or colname not in df_use:
        return np.nan
    s = pd.to_numeric(df_use[colname], errors="coerce")
    return float(s.mean()) if s.notna().any() else np.nan

kpi_company_followers = safe_mean(key["company_followers"])
kpi_employees = safe_mean(key["employees"])
kpi_followers = safe_mean(key["followers"])

col1, col2, col3, col4 = st.columns(4)
col1.metric("Leads (total)", f"{total_leads:,}")
col2.metric("Entreprises uniques", f"{int(unique_companies):,}" if pd.notna(unique_companies) else "—")
col3.metric("Followers (moy.)", f"{kpi_followers:,.0f}" if pd.notna(kpi_followers) else "—")
col4.metric("Employés (moy.)", f"{kpi_employees:,.0f}" if pd.notna(kpi_employees) else "—")

# ---------------- Top filters (numeric/date/categorical spécifiques)
st.markdown("### 🎛️ Filtres principaux")
top_filters = {}

# Numeric sliders helper
def add_numeric_slider(label, series, key_name):
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().any():
        lo, hi = float(np.nanmin(s)), float(np.nanmax(s))
        if lo != hi:
            v = st.slider(label, lo, hi, (lo, hi))
            top_filters[key_name] = ("num_range", v)

# Categorical multiselect helper
def add_categorical_multiselect(label, series, key_name, max_unique=20):
    uniques = series.dropna().astype(str).unique()
    if 1 <= len(uniques) <= max_unique:
        v = st.multiselect(label, sorted(map(str, uniques)))
        if v:
            top_filters[key_name] = ("in", set(v))

cols_top = st.columns(3)
with cols_top[0]:
    if key["company_followers"]:
        add_numeric_slider(prettify_label(key["company_followers"]), df_use[key["company_followers"]], key["company_followers"])
    if key["followers"]:
        add_numeric_slider(prettify_label(key["followers"]), df_use[key["followers"]], key["followers"])
with cols_top[1]:
    if key["connections"]:
        add_numeric_slider(prettify_label(key["connections"]), df_use[key["connections"]], key["connections"])
    if key["employees"]:
        add_numeric_slider(prettify_label(key["employees"]), df_use[key["employees"]], key["employees"])
with cols_top[2]:
    if key["company_size"]:
        add_categorical_multiselect(prettify_label(key["company_size"]), df_use[key["company_size"]], key["company_size"], max_unique=50)
    if key["company_founded"]:
        # treat founded year as numeric
        add_numeric_slider(prettify_label(key["company_founded"]), df_use[key["company_founded"]], key["company_founded"])

# ---------------- Sidebar text search filters
st.sidebar.header("🔎 Recherche (texte)")
text_filters = {}
# Choose a set of candidate text fields to search in (excluding ignored and numeric-heavy)
candidates_text = []
for c in df_use.columns:
    if c == key.get("photo"):
        continue
    # Skip numeric-like columns
    if pd.api.types.is_numeric_dtype(df_use[c]):
        continue
    # Skip datetime columns
    if pd.api.types.is_datetime64_any_dtype(df_use[c]):
        continue
    # Skip ignored semantics already covered above
    if c in (key.get("company_followers"), key.get("followers"), key.get("connections"),
             key.get("employees"), key.get("company_size"), key.get("company_founded")):
        continue
    candidates_text.append(c)

with st.sidebar.expander("Choisir les colonnes à rechercher", expanded=False):
    selected_text_cols = st.multiselect("Colonnes texte", options=[prettify_label(c) for c in candidates_text],
                                        default=[prettify_label(c) for c in candidates_text[:6]])
# Map pretty -> real
pretty_to_real = {prettify_label(c): c for c in candidates_text}
selected_real = [pretty_to_real[p] for p in selected_text_cols if p in pretty_to_real]

for c in selected_real:
    v = st.sidebar.text_input(f"{prettify_label(c)}", "")
    if v:
        text_filters[c] = ("contains", v.lower())

# ---------------- Apply filters
def apply_all_filters(df_in: pd.DataFrame):
    df_out = df_in.copy()
    mask = pd.Series([True] * len(df_out), index=df_out.index)
    # top numeric/cat filters
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
    # text filters
    for col_key, (ftype, val) in text_filters.items():
        if col_key not in df_out:
            continue
        s = df_out[col_key]
        mask &= s.astype(str).str.lower().str.contains(val, na=False)
    return df_out[mask]

filtered = apply_all_filters(df_use)

# ---------------- Pagination
st.sidebar.header("🧭 Pagination")
page_size = st.sidebar.selectbox("Taille de page", [10, 25, 50, 100], index=1)
total_rows = len(filtered)
total_pages = (total_rows - 1) // page_size + 1 if total_rows > 0 else 1
page = st.sidebar.number_input("Page", min_value=1, max_value=max(1, total_pages), value=1, step=1)
start, end = (page - 1) * page_size, (page - 1) * page_size + page_size

st.markdown(f"**{total_rows:,} leads** après filtres • Page **{page}/{total_pages}**")

# ---------------- Card view
st.markdown("### 🧾 Résultats")
subset = filtered.iloc[start:end]

# Prepare fields
col_first = key["first_name"]
col_last = key["last_name"]
col_full = key["full_name"]
col_company = key["company"]
col_job = key["job"]
col_photo = key["photo"]
col_location = key["location"]

def display_name(row):
    if col_full and pd.notna(row.get(col_full, np.nan)) and str(row.get(col_full)).strip():
        return str(row[col_full]).strip()
    parts = []
    if col_first and pd.notna(row.get(col_first, np.nan)):
        parts.append(str(row[col_first]).strip())
    if col_last and pd.notna(row.get(col_last, np.nan)):
        parts.append(str(row[col_last]).strip())
    return " ".join(parts) if parts else "(Sans nom)"

for _, r in subset.iterrows():
    with st.container():
        c1, c2 = st.columns([1, 5])
        with c1:
            if col_photo and pd.notna(r.get(col_photo, np.nan)):
                try:
                    st.image(r[col_photo], use_container_width=True)
                except Exception:
                    st.write("")
            else:
                st.write("")
        with c2:
            st.markdown(f"**{display_name(r)}**")
            line2 = []
            if col_job and pd.notna(r.get(col_job, np.nan)):
                line2.append(str(r[col_job]))
            if col_company and pd.notna(r.get(col_company, np.nan)):
                line2.append(f"@ {r[col_company]}")
            if line2:
                st.write(" — ".join(line2))
            if col_location and pd.notna(r.get(col_location, np.nan)):
                st.caption(str(r[col_location]))

        st.divider()

# ---------------- Export & Stats table
st.download_button("⬇️ Télécharger le CSV filtré", data=filtered.to_csv(index=False).encode("utf-8"),
                   file_name="leads_filtres.csv", mime="text/csv")

with st.expander("📊 Statistiques détaillées"):
    # Compat describe
    try:
        st.write(filtered.describe(include="all", datetime_is_numeric=True))
    except TypeError:
        st.write(filtered.describe(include="all"))
