
import streamlit as st
import pandas as pd
import numpy as np

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
st.caption("Liste + filtres + pagination (emails obligatoires)")

uploaded = st.file_uploader("Déposez votre fichier CSV", type=["csv"])
if uploaded is None:
    st.info("Déposez un fichier CSV pour commencer.")
    st.stop()

df = read_csv_any(uploaded)

# --- Colonnes principales
col_first = find_col(df, ["firstName", "first_name", "firstname", "given name", "givenName"])
col_last = find_col(df, ["lastName", "last_name", "lastname", "family name", "surname"])
col_company = find_col(df, ["companyName", "company name", "company", "employer"])
col_job = find_col(df, ["linkedinHeadline", "job title", "title", "headline", "position", "role"])
col_location = find_col(df, ["linkedinJobLocation", "location", "city", "country", "region"])
col_email = find_col(df, ["email", "mail", "emailaddress", "contact email"])

# --- Filtre email obligatoire
if col_email:
    df = df[df[col_email].astype(str).str.contains("@", na=False)]
else:
    st.error("Aucune colonne email détectée (email/mail).")
    st.stop()

# --- Stats
st.markdown("### 📈 Statistiques")
c1, c2 = st.columns(2)
c1.metric("Leads (emails valides)", f"{len(df):,}")
if col_company:
    c2.metric("Entreprises uniques", f"{df[col_company].nunique():,}")

# --- Filtres détectés
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

# === Sidebar: Pagination en HAUT + navigation
st.sidebar.header("🧭 Pagination")
page_size_choice = st.sidebar.selectbox("Taille de page", [12, 24, 48, 96, "TOUT"], index=1, key="page_size_choice")
if "page_num" not in st.session_state:
    st.session_state.page_num = 1


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

# Build the list of available filters in desired order
_filter_specs = []
if followers_col:
    _filter_specs.append(("Nombre de followers", followers_col, "num"))
if connections_col:
    _filter_specs.append(("Nombre de connexions", connections_col, "num"))
if company_size_col:
    _filter_specs.append(("Taille de l'entreprise", company_size_col, "cat"))
if company_founded_col:
    _filter_specs.append(("Création de l'entreprise", company_founded_col, "num"))

cols_top = st.columns(3, gap="large")
for idx, (label, colname, kind) in enumerate(_filter_specs):
    with cols_top[idx % 3]:
        if kind == "num":
            add_numeric_slider(label, df[colname], colname)
        else:
            add_categorical_multiselect(label, df[colname], colname, max_unique=50)

# --- Sidebar: Filtres texte

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

# --- Calcul pagination
total_rows = len(filtered)
if page_size_choice == "TOUT":
    page_size = total_rows if total_rows > 0 else 1
    total_pages = 1
    st.session_state.page_num = 1
else:
    page_size = int(page_size_choice)
    total_pages = (total_rows - 1) // page_size + 1 if total_rows > 0 else 1

# Prev / Next buttons
cprev, cpage, cnext = st.sidebar.columns([1, 2, 1])
with cprev:
    if st.button("◀", disabled=(st.session_state.page_num <= 1)):
        st.session_state.page_num = max(1, st.session_state.page_num - 1)
with cpage:
    st.number_input("Page", min_value=1, max_value=max(1, total_pages), value=st.session_state.page_num, step=1, key="page_num_input")
    st.session_state.page_num = st.session_state.page_num_input
with cnext:
    if st.button("▶", disabled=(st.session_state.page_num >= total_pages)):
        st.session_state.page_num = min(total_pages, st.session_state.page_num + 1)

page = st.session_state.page_num
start, end = (page - 1) * page_size, (page - 1) * page_size + page_size
st.caption(f"{total_rows:,} leads après filtres • Page {page}/{total_pages}")

# --- UI List stylée
st.markdown("### 🧾 Résultats")
st.markdown(
    """
    <style>
    .list-header, .list-row {
        display: grid;
        grid-template-columns: 28% 42% 20% 10%;
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
    """,
    unsafe_allow_html=True,
)

def safe_str(x):
    return "" if pd.isna(x) else str(x)

def render_row(row):
    first = safe_str(row.get(col_first)) if col_first else ""
    last = safe_str(row.get(col_last)) if col_last else ""
    name = (first + " " + last).strip() or "(Sans nom)"
    company = safe_str(row.get(col_company)) if col_company else ""
    job = safe_str(row.get(col_job)) if col_job else ""
    company_job = " — ".join([s for s in [company, job] if s])
    email = safe_str(row.get(col_email))
    email_html = f'<a href="mailto:{email}">{email}</a>' if "@" in email else ""
    loc = safe_str(row.get(col_location)) if col_location else ""
    html = f'''
    <div class="list-row">
        <div><strong>{name}</strong></div>
        <div>{company_job}</div>
        <div>{email_html}</div>
        <div class="muted">{loc}</div>
    </div>
    '''
    st.markdown(html, unsafe_allow_html=True)

# Header
st.markdown(
    '''
    <div class="list-header">
        <div>Nom</div><div>Entreprise — Poste</div><div>Email</div><div>Localisation</div>
    </div>
    ''',
    unsafe_allow_html=True
)

subset = filtered.iloc[start:end]
for _, r in subset.iterrows():
    render_row(r)

# --- Export CSV
st.download_button(
    "⬇️ Télécharger le CSV filtré",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="leads_filtres.csv",
    mime="text/csv",
)
