import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any

st.set_page_config(page_title="Lead Manager", page_icon="📇", layout="wide")

# ---------- Helpers
def prettify_label(s: str) -> str:
    if not isinstance(s, str):
        return str(s)
    s = s.replace("_", " ").replace("-", " ")
    for k in ["linkedin", "LinkedIn", "Linkedin"]:
        s = s.replace(k, "")
    s = " ".join(s.split())
    return s.strip().title()

# French labels mapping for column names
FRENCH_LABELS = {
    "companyindustry": "Secteur d’activité",
    "companyname": "Nom de l’entreprise",
    "companywebsite": "Site web de l’entreprise",
    "companyurl": "URL de l’entreprise",
    "companyslug": "Identifiant entreprise (slug)",
    "companyheadquarter": "Siège social",
    "companyspecialities": "Spécialités de l’entreprise",
    "companydescription": "Description de l’entreprise",
    "companytagline": "Slogan / Baseline",
    "firstname": "Prénom",
    "lastname": "Nom",
    "scraperfullname": "Nom complet",
    "profileslug": "Identifiant profil (slug)",
    "profileurl": "URL du profil",
    "profileurn": "Identifiant LinkedIn (URN)",
    "profileimageurl": "Photo de profil",
    "profileimageurn": "Identifiant photo (URN)",
    "professionalemail": "Email professionnel",
    "refreshedat": "Date de mise à jour",
    "mutualconnectionsurl": "URL des relations communes",
    "connectionsurl": "URL des connexions",
    "headline": "Titre / Fonction actuelle",
    "ishiringbadge": "Recrute actuellement",
    "isopentoworkbadge": "Ouvert aux opportunités",
    "jobdaterange": "Période d’emploi actuelle",
    "joblocation": "Lieu de travail",
    "jobtitle": "Poste actuel",
    "jobdescription": "Description du poste",
    "previouscompanyname": "Ancienne entreprise",
    "previouscompanyslug": "Identifiant ancienne entreprise",
    "previousjobdaterange": "Période d’emploi précédente",
    "previousjoblocation": "Lieu de l’emploi précédent",
    "previousjobtitle": "Poste précédent",
    "previousjobdescription": "Description du poste précédent",
    "schoolname": "Établissement scolaire",
    "schoolurl": "Site de l’école",
    "schoolcompanyslug": "Identifiant de l’école",
    "schooldaterange": "Période d’études",
    "schooldegree": "Diplôme obtenu",
    "schooldescription": "Description de la formation",
    "previousschoolname": "Ancienne école",
    "previousschoolurl": "Site de l’ancienne école",
    "previousschoolcompanyslug": "Identifiant ancienne école",
    "previousschooldaterange": "Période ancienne formation",
    "previousschooldegree": "Diplôme ancien",
    "previousschooldescription": "Description ancienne formation",
    "skillslabel": "Compétences",
    "location": "Localisation générale",
    "description": "Description du profil",
}

def fr_label(col: str) -> str:
    if not isinstance(col, str):
        return str(col)
    key = col.lower()
    return FRENCH_LABELS.get(key, prettify_label(col))

def find_col(df: pd.DataFrame, candidates) -> Optional[str]:
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
def read_any(uploaded, sheet_name: Optional[str]) -> Dict[str, Any]:
    """Return dict with 'df' and 'sheets' (if Excel)."""
    name = uploaded.name.lower()
    out: Dict[str, Any] = {"df": None, "sheets": None}
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded, sep=None, engine="python", low_memory=False)
            out["df"] = df
        elif name.endswith(".xlsx") or name.endswith(".xls"):
            xls = pd.ExcelFile(uploaded)
            out["sheets"] = xls.sheet_names
            use_sheet = sheet_name or xls.sheet_names[0]
            df = pd.read_excel(xls, sheet_name=use_sheet, dtype=str)
            out["df"] = df
        else:
            df = pd.read_csv(uploaded, low_memory=False)
            out["df"] = df
    except Exception as e:
        try:
            df = pd.read_excel(uploaded, dtype=str)
            out["df"] = df
        except Exception:
            raise e
    return out

# ---------- UI
st.title("📇 Lead Manager")
st.caption("Excel/CSV → liste + filtres + pagination (emails obligatoires)")

uploaded = st.file_uploader("Déposez votre fichier CSV ou Excel", type=["csv", "xlsx", "xls"])
if uploaded is None:
    st.info("Déposez un fichier pour commencer.")
    st.stop()

# Excel → choix de la feuille
chosen_sheet = None
if uploaded.name.lower().endswith((".xlsx", ".xls")):
    probe = read_any(uploaded, sheet_name=None)
    sheets = probe.get("sheets") or []
    if sheets:
        chosen_sheet = st.selectbox("Feuille Excel", sheets, index=0)
    df = probe["df"] if chosen_sheet is None else read_any(uploaded, sheet_name=chosen_sheet)["df"]
else:
    df = read_any(uploaded, sheet_name=None)["df"]

# --- Colonnes principales (détection souple)
col_first = find_col(df, ["firstName", "first_name", "firstname", "given name", "givenName"])
col_last = find_col(df, ["lastName", "last_name", "lastname", "family name", "surname"])
col_fullname = find_col(df, ["Nom complet", "scraperfullname", "full name", "fullname"])
col_company = find_col(df, ["companyName", "company name", "company", "employer"])
col_job = find_col(df, ["linkedinHeadline", "job title", "title", "headline", "position", "role"])
col_location = find_col(df, ["linkedinJobLocation", "location", "city", "country", "region"])
col_email = find_col(df, ["professionalemail", "email", "mail", "emailaddress", "contact email"])
col_phone = find_col([...,"Téléphone","telephone","phone","mobile","phonenumber", "Téléphone Société"])

# --- Filtre email obligatoire
if col_email:
    df = df[df[col_email].astype(str).str.contains("@", na=False)]
else:
    st.error("Aucune colonne email détectée (ex. ProfessionalEmail, Email, Mail).")
    st.stop()

# --- Stats
st.markdown("### 📈 Statistiques")
c1, c2 = st.columns(2)
c1.metric("Leads (emails valides)", f"{len(df):,}")
if col_company:
    c2.metric("Entreprises uniques", f"{df[col_company].nunique():,}")

# --- Colonnes utilisées pour les filtres avancés (si présentes)
followers_col = find_col(df, ["followers", "followerscount"])
connections_col = find_col(df, ["connections", "connection", "connexion"])
company_size_col = find_col(df, ["companysize", "company size", "size"])
company_founded_col = find_col(df, ["companyfounded", "founded", "foundation year"])
last_revenue_col = find_col([...,"Dernier CA Publié", "dernier ca publie", "chiffre d'affaires"])
last_result_col = find_col([...,"Dernier Résultat Publié", "dernier resultat publie"])

# === Sidebar: Pagination (en haut) + navigation
st.sidebar.header("🧭 Pagination")
page_size_choice = st.sidebar.selectbox("Taille de page", [12, 24, 48, 96, "TOUT"], index=1, key="page_size_choice")
if "page_num" not in st.session_state:
    st.session_state.page_num = 1

# --- Filtres principaux (rangées de 3)
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

filter_specs = []
if followers_col:
    filter_specs.append(("Nombre de followers", followers_col, "num"))
if connections_col:
    filter_specs.append(("Nombre de connexions", connections_col, "num"))
if company_size_col:
    filter_specs.append(("Taille de l'entreprise", company_size_col, "cat"))
if company_founded_col:
    filter_specs.append(("Création de l'entreprise", company_founded_col, "num"))

for i in range(0, len(filter_specs), 3):
    row = filter_specs[i:i+3]
    cols_row = st.columns(3, gap="large")
    for j, (label, colname, kind) in enumerate(row):
        with cols_row[j]:
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
    val = st.sidebar.text_input(fr_label(c), "", key=f"txt_{c}")
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

# --- Pagination (calcul)
total_rows = len(filtered)
if page_size_choice == "TOUT":
    page_size = total_rows if total_rows > 0 else 1
    total_pages = 1
    st.session_state.page_num = 1
else:
    page_size = int(page_size_choice)
    total_pages = (total_rows - 1) // page_size + 1 if total_rows > 0 else 1

# Prev / Next
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

# --- Liste stylée
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
    name = safe_str(row.get(col_fullname)) if col_last else ""
    company = safe_str(row.get(col_company)) if col_company else ""
    job = safe_str(row.get(col_job)) if col_job else ""
    telephone = safe_str(row.get(col_phone)) if col_job else ""
    company_job = " — ".join([s for s in [company, job] if s])
    email = safe_str(row.get(col_email))
    email_html = f'<a href="mailto:{email}">{email}</a>' if "@" in email else ""
    loc = safe_str(row.get(col_location)) if col_location else ""
    html = f'''
    <div class="list-row">
        <div><strong>{name}</strong></div>
        <div>{company_job}</div>
        <div>{email_html}</div>
        <div>{telephone}</div>
        <div class="muted">{loc}</div>
    </div>
    '''
    st.markdown(html, unsafe_allow_html=True)

# En-tête
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
