import streamlit as st
import pandas as pd
import numpy as np
import io
import requests
from urllib.parse import quote

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


# ---------- Interface
st.title("📇 Lead Manager")
st.caption("Affichage des leads en cartes avec images LinkedIn.")

uploaded = st.file_uploader("Déposez votre fichier CSV", type=["csv"])
if uploaded is None:
    st.info("Déposez un fichier CSV pour commencer.")
    st.stop()

df = read_csv_any(uploaded)

# --- Détection des colonnes
col_first = "firstName" if "firstName" in df.columns else None
col_last = "lastName" if "lastName" in df.columns else None
col_company = "companyName" if "companyName" in df.columns else None
col_job = "linkedinHeadline" if "linkedinHeadline" in df.columns else None
col_photo = "linkedinProfileImageUrl" if "linkedinProfileImageUrl" in df.columns else None
col_location = "linkedinJobLocation" if "linkedinJobLocation" in df.columns else None

st.success(f"Colonnes détectées : photo → {col_photo}, entreprise → {col_company}, job → {col_job}")

# --- Statistiques
col1, col2 = st.columns(2)
col1.metric("Leads", f"{len(df):,}")
if col_company:
    col2.metric("Entreprises uniques", f"{df[col_company].nunique():,}")

# --- Résultats sous forme de cartes
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
    """,
    unsafe_allow_html=True,
)

cards_per_row = st.selectbox("Cartes par ligne", [3, 4, 6], index=1)

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
    if col_photo and pd.notna(rec.get(col_photo, None)):
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


# Grille de cartes
rows = []
recs = list(df.to_dict(orient="records"))
for i in range(0, len(recs), cards_per_row):
    rows.append(recs[i:i + cards_per_row])

for row in rows:
    cols = st.columns(cards_per_row)
    for col, rec in zip(cols, row):
        with col:
            render_card(rec)

# --- Debug
with st.expander("🔧 Debug images"):
    if col_photo and col_photo in df.columns:
        urls = df[col_photo].dropna().astype(str).head(5).tolist()
        st.write("Exemples d’URLs :", urls)
        for u in urls:
            st.image(weserv_proxy(u))
    else:
        st.warning("Aucune colonne image détectée.")
