
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

def find_col(df: pd.DataFrame, candidates) -> str | None:
    if df is None or df.empty:
        return None
    cmap = {c.lower(): c for c in df.columns}
    # exact match
    for cand in candidates:
        lc = cand.lower()
        if lc in cmap:
            return cmap[lc]
    # contains
    for cand in candidates:
        lc = cand.lower()
        for k, real in cmap.items():
            if lc in k:
                return real
    return None

@st.cache_data(show_spinner=False)
def read_csv_any(uploaded) -> pd.DataFrame:
    try:
        return pd.read_csv(uploaded, sep=None, engine="python", low_memory=False)
    except Exception:
        uploaded.seek(0)
        return pd.read_csv(uploaded, low_memory=False)

# ---------- UI
st.title("📇 Lead Manager")
st.caption("Liste simple + filtres + pagination (emails obligatoires)")

uploaded = st.file_uploader("Déposez votre fichier CSV", type=["csv"])
if uploaded is None:
    st.info("Déposez un fichier CSV pour commencer.")
    st.stop()

df = read_csv_any(uploaded)

# --- Colonnes principales (détection robuste)
col_first = find_col(df, ["firstName", "first_name", "firstname", "given name", "givenName"])
col_last = find_col(df, ["lastName", "last_name", "lastname", "family name", "surname"])
col_company = find_col(df, ["companyName", "company name", "company", "employer"])
col_job = find_col(df, ["linkedinHeadline", "job title", "title", "headline", "position", "role"])
col_location = find_col(df, ["linkedinJobLocation", "location", "city", "country", "region"])
col_email = find_col(df, ["email", "mail", "emailaddress", "contact email"])

# --- Filtre automatique: ne garder que les lignes avec un email valide
if col_email:
    df = df[df[col_email].astype(str).str.contains("@", na=False)]
else:
    st.error("Aucune colonne email détectée (email/mail). Impossible de continuer sans email.")
    st.stop()

# --- Stats
st.markdown("### 📈 Statistiques")
c1, c2 = st.columns(2)
c1.metric("Leads (emails valides)", f"{len(df):,}")
if col_company:
    c2.metric("Entreprises uniques", f"{df[col_company].nunique():,}")

# --- Détection colonnes pour filtres principaux
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

# === Sidebar: Pagination en HAUT ===
st.sidebar.header("🧭 Pagination")
default_page_size = 24
page_size = st.sidebar.selectbox("Taille de page", [12, 24, 48, 96], index=[12,24,48,96].index(default_page_size), key="page_size")

# --- Filtres principaux (3 colonnes, en page principale)
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
i = 0
if followers_col:
    with cols_top[i % 3]:
        add_numeric_slider("Nombre de followers", df[followers_col], followers_col)
    i += 1
if connections_col:
    with cols_top[i % 3]:
        add_numeric_slider("Nombre de connexions", df[connections_col], connections_col)
    i += 1
if company_size_col:
    with cols_top[i % 3]:
        add_categorical_multiselect("Taille de l'entreprise", df[company_size_col], company_size_col, max_unique=50)
    i += 1
if company_founded_col:
    with cols_top[i % 3]:
        add_numeric_slider("Création de l'entreprise", df[company_founded_col], company_founded_col)
    i += 1

# --- Sidebar: Filtres texte (sous la pagination)
st.sidebar.header("🔎 Recherche (texte)")
text_filters = {}
skip_cols = {x for x in [followers_col, connections_col, company_size_col, company_founded_col] if x}
for c in df.columns:
    if c in skip_cols:
        continue
    if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_datetime64_any_dtype(df[c]):
        continue
    val = st.sidebar.text_input(prettify_label(c), "", key=f"txt_{c}")
    if val:
        text_filters[c] = ("contains", val.lower())

# --- Appliquer les filtres
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
            rng = s_num.between(lo, hi)
            mask &= (rng | s_num.isna())
        elif ftype == "in":
            mask &= s.astype(str).isin(val)
    for col_key, (ftype, val) in text_filters.items():
        if col_key not in df_out:
            continue
        s = df_out[col_key]
        mask &= s.astype(str).str.lower().str.contains(val, na=False)
    return df_out[mask]

filtered = apply_all_filters(df)

# --- Calcul pagination (après filtres)
total_rows = len(filtered)
total_pages = (total_rows - 1) // page_size + 1 if total_rows > 0 else 1
page = st.sidebar.number_input("Page", min_value=1, max_value=max(1, total_pages), value=1, step=1, key="page_num")
start, end = (page - 1) * page_size, (page - 1) * page_size + page_size
st.caption(f"{total_rows:,} leads après filtres • Page {page}/{total_pages}")

# --- Affichage en liste
st.markdown("### 🧾 Résultats")
subset = filtered.iloc[start:end].copy()

def line_for_row(row):
    # First Last
    parts = []
    if col_first and pd.notna(row.get(col_first, None)):
        parts.append(str(row[col_first]).strip())
    if col_last and pd.notna(row.get(col_last, None)):
        parts.append(str(row[col_last]).strip())
    name = " ".join(parts) if parts else "(Sans nom)"
    # Company - Job
    company = str(row[col_company]).strip() if col_company and pd.notna(row.get(col_company, None)) else ""
    job = str(row[col_job]).strip() if col_job and pd.notna(row.get(col_job, None)) else ""
    company_job = f"{company} - {job}" if (company or job) else ""
    # Email
    email = str(row[col_email]).strip() if col_email and pd.notna(row.get(col_email, None)) else ""
    # Location
    loc = str(row[col_location]).strip() if col_location and pd.notna(row.get(col_location, None)) else ""
    return f"{name} | {company_job} | {email} | {loc}"

for _, r in subset.iterrows():
    st.write(line_for_row(r))

# --- Export CSV
st.download_button(
    "⬇️ Télécharger le CSV filtré",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="leads_filtres.csv",
    mime="text/csv",
)

st.markdown("""
    <style>
    .list-header, .list-row {
        display: grid;
        grid-template-columns: 1.2fr 2fr 1.6fr 1.2fr;
        gap: 12px;
        align-items: baseline;
        padding: 10px 12px;
    }
    .list-header {
        border-bottom: 2px solid rgba(0,0,0,0.08);
        font-weight: 700;
        color: rgba(49,51,63,0.8);
        margin-bottom: 4px;
    }
    .list-row {
        border-bottom: 1px solid rgba(0,0,0,0.06);
    }
    .muted { color: rgba(49,51,63,0.6); }
    a { text-decoration: none; }
    </style>
""", unsafe_allow_html=True)
