
import streamlit as st
import pandas as pd
import io
from datetime import datetime

st.set_page_config(page_title="Lead Manager", page_icon="📇", layout="wide")

@st.cache_data(show_spinner=False)
def read_csv_any(file) -> pd.DataFrame:
    try:
        return pd.read_csv(file, sep=None, engine="python", low_memory=False)
    except Exception:
        file.seek(0)
        return pd.read_csv(file, low_memory=False)

def coerce_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    # Try to coerce any column that looks like a date
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().head(50).astype(str)
            # Heuristic: contains date-like characters
            if sample.str.contains(r"\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{2,4}", regex=True).mean() > 0.5:
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
                except Exception:
                    pass
    return df

def build_filters(df: pd.DataFrame):
    st.sidebar.header("🔎 Filtres")
    filters = {}

    # Let user pick which columns to filter (optional)
    with st.sidebar.expander("Choisir des colonnes à filtrer", expanded=False):
        selected_cols = st.multiselect("Colonnes", options=list(df.columns), default=list(df.columns))
    work_df = df[selected_cols].copy() if selected_cols else df.copy()

    for col in work_df.columns:
        col_series = work_df[col]
        if pd.api.types.is_numeric_dtype(col_series):
            min_val, max_val = float(col_series.min()), float(col_series.max())
            if min_val == max_val:
                continue
            v = st.sidebar.slider(f"{col} (min-max)", min_val, max_val, (min_val, max_val))
            filters[col] = ("range", v)
        elif pd.api.types.is_datetime64_any_dtype(col_series):
            min_date, max_date = col_series.min(), col_series.max()
            if pd.isna(min_date) or pd.isna(max_date) or min_date == max_date:
                continue
            v = st.sidebar.date_input(f"{col} (période)", (min_date.date(), max_date.date()))
            if isinstance(v, tuple) and len(v) == 2:
                filters[col] = ("daterange", v)
        else:
            # Categorical vs free text decision by cardinality
            uniques = col_series.dropna().unique()
            if len(uniques) > 0 and len(uniques) <= 50:
                v = st.sidebar.multiselect(f"{col}", sorted(map(str, uniques)))
                if v:
                    filters[col] = ("in", set(map(str, v)))
            else:
                v = st.sidebar.text_input(f"Recherche texte dans {col}", "")
                if v:
                    filters[col] = ("contains", v.lower())

    return filters, work_df

def apply_filters(df: pd.DataFrame, filters):
    if not filters:
        return df
    mask = pd.Series([True] * len(df), index=df.index)
    for col, (ftype, val) in filters.items():
        if col not in df.columns:
            continue
        s = df[col]
        if ftype == "range" and pd.api.types.is_numeric_dtype(s):
            lo, hi = val
            mask &= s.astype(float).between(lo, hi)
        elif ftype == "daterange" and pd.api.types.is_datetime64_any_dtype(s):
            start, end = val
            mask &= s.between(pd.to_datetime(start), pd.to_datetime(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
        elif ftype == "in":
            mask &= s.astype(str).isin(val)
        elif ftype == "contains":
            mask &= s.astype(str).str.lower().str.contains(val, na=False)
    return df[mask]

st.title("📇 Lead Manager — CSV → Dashboard")
st.caption("Glissez-déposez un fichier CSV pour explorer vos leads avec des filtres dynamiques, pagination et export.")

uploaded = st.file_uploader("Déposez votre CSV ici", type=["csv"], accept_multiple_files=False)
if uploaded is None:
    st.info("Aucun fichier importé. Un petit échantillon interne sera utilisé si disponible.")
    st.stop()

df = read_csv_any(uploaded)
# Option to coerce datetimes
with st.expander("Options d'import", expanded=False):
    if st.checkbox("Tenter de détecter les colonnes de type date", value=True):
        df = coerce_datetimes(df)
    # Allow user to strip spaces in columns
    if st.checkbox("Supprimer les espaces au début/fin des cellules texte", value=False):
        df = df.apply(lambda col: col.str.strip() if col.dtype == object else col)

st.success(f"{len(df):,} lignes • {len(df.columns)} colonnes")

# Build and apply filters
filters, work_df = build_filters(df)
filtered = apply_filters(df, filters)

# Pagination controls
st.sidebar.header("🧭 Pagination")
page_size = st.sidebar.selectbox("Taille de page", [10, 25, 50, 100, 200], index=1)
total_rows = len(filtered)
total_pages = (total_rows - 1) // page_size + 1 if total_rows > 0 else 1
page = st.sidebar.number_input("Page", min_value=1, max_value=max(1, total_pages), value=1, step=1)

start = (page - 1) * page_size
end = start + page_size

st.subheader("📄 Résultats filtrés")
st.caption(f"{total_rows:,} lignes après filtre • Page {page}/{total_pages}")
st.dataframe(filtered.iloc[start:end], use_container_width=True)

# Download filtered CSV
csv_bytes = filtered.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Télécharger le CSV filtré", data=csv_bytes, file_name="leads_filtrés.csv", mime="text/csv")

# Quick stats
with st.expander("Statistiques rapides"):
    st.write(filtered.describe(include="all", datetime_is_numeric=True))
