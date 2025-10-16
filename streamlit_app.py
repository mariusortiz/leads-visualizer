
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
        try:
            file.seek(0)
        except Exception:
            pass
        return pd.read_csv(file, low_memory=False)

def coerce_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    # Parse date-looking object columns and drop timezone info to avoid comparison errors
    for col in df.columns:
        try:
            if df[col].dtype == object:
                sample = df[col].dropna().head(50).astype(str)
                if sample.str.contains(r"\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{2,4}", regex=True).mean() > 0.5:
                    df[col] = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True, utc=False)
            # If already tz-aware, make it tz-naive
            if pd.api.types.is_datetime64tz_dtype(df[col]):
                df[col] = df[col].dt.tz_localize(None)
        except Exception:
            pass
    return df

def build_filters(df: pd.DataFrame):
    st.sidebar.header("🔎 Filtres")
    filters = {}

    with st.sidebar.expander("Choisir des colonnes à filtrer", expanded=False):
        selected_cols = st.multiselect("Colonnes", options=list(df.columns), default=list(df.columns))
    work_df = df[selected_cols].copy() if selected_cols else df.copy()

    for col in work_df.columns:
        col_series = work_df[col]
        if pd.api.types.is_numeric_dtype(col_series):
            try:
                min_val, max_val = float(pd.to_numeric(col_series, errors="coerce").min()), float(pd.to_numeric(col_series, errors="coerce").max())
            except Exception:
                continue
            if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
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
            uniques = col_series.dropna().astype(str).unique()
            if 0 < len(uniques) <= 50:
                v = st.sidebar.multiselect(f"{col}", sorted(map(str, uniques)))
                if v:
                    filters[col] = ("in", set(map(str, v)))
            else:
                v = st.sidebar.text_input(f"Recherche texte dans {col}", "")
                if v:
                    filters[col] = ("contains", v.lower())

    return filters, work_df

def apply_filters(df: pd.DataFrame, filters):
    if not filters or df.empty:
        return df
    mask = pd.Series([True] * len(df), index=df.index)
    for col, (ftype, val) in filters.items():
        if col not in df.columns:
            continue
        s = df[col]
        if ftype == "range":
            s_num = pd.to_numeric(s, errors="coerce")
            lo, hi = val
            mask &= s_num.between(lo, hi)
        elif ftype == "daterange" and pd.api.types.is_datetime64_any_dtype(s):
            start, end = val
            start = pd.to_datetime(start)
            end = pd.to_datetime(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            mask &= s.between(start, end)
        elif ftype == "in":
            mask &= s.astype(str).isin(val)
        elif ftype == "contains":
            mask &= s.astype(str).str.lower().str.contains(val, na=False)
    return df[mask]

st.title("📇 Lead Manager — CSV → Dashboard")
st.caption("Glissez-déposez un fichier CSV pour explorer vos leads avec des filtres dynamiques, pagination et export.")

uploaded = st.file_uploader("Déposez votre CSV ici", type=["csv"], accept_multiple_files=False)
if uploaded is None:
    st.info("Aucun fichier importé. Déposez un CSV pour commencer.")
    st.stop()

df = read_csv_any(uploaded)

with st.expander("Options d'import", expanded=False):
    if st.checkbox("Tenter de détecter les colonnes de type date", value=True):
        df = coerce_datetimes(df)
    if st.checkbox("Supprimer les espaces au début/fin des cellules texte", value=False):
        df = df.apply(lambda col: col.str.strip() if col.dtype == object else col)

st.success(f"{len(df):,} lignes • {len(df.columns)} colonnes")

filters, work_df = build_filters(df)
filtered = apply_filters(df, filters)

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

csv_bytes = filtered.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Télécharger le CSV filtré", data=csv_bytes, file_name="leads_filtres.csv", mime="text/csv")

with st.expander("Statistiques rapides"):
    # pandas compatibility: datetime_is_numeric may not exist in older versions
    try:
        st.write(filtered.describe(include="all", datetime_is_numeric=True))
    except TypeError:
        st.write(filtered.describe(include="all"))
