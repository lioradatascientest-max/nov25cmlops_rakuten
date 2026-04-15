import os

from components.auth import is_admin, is_authenticated, login_form, logout
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Rakuten MLOps",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

if not is_authenticated():
    login_form()
    st.stop()

with st.sidebar:
    st.markdown("### Rakuten MLOps")
    st.caption(f"Connecté : **{st.session_state['username']}**")
    role = st.session_state.get("role", "user")
    badge = "🔴 admin" if role == "admin" else "🟢 user"
    st.caption(badge)
    st.divider()
    if is_admin():
        st.page_link("pages/1_predict.py",    label="Prédiction",  icon="🔍")
        st.page_link("pages/2_pipeline.py",   label="Pipeline",    icon="⚙️")
        st.page_link("pages/3_monitoring.py", label="Monitoring",  icon="📊")
        st.page_link("pages/4_presentation.py", label="Présentation", icon="🎤")
    else:
        st.page_link("pages/1_predict.py", label="Prédiction", icon="🔍")
    st.divider()
    if st.button("Déconnexion", use_container_width=True):
        logout()

# ── Page d'accueil ────────────────────────────────────────────────────────────
st.title("🛍️ Rakuten MLOps — Dashboard")
st.caption(
    "Plateforme de classification automatique de produits e-commerce · Rakuten France")

st.divider()

# ── Vue USER ──────────────────────────────────────────────────────────────────
if not is_admin():
    st.subheader("Bienvenue 👋")
    st.markdown("""
    Cette plateforme permet de **classifier automatiquement des produits Rakuten**
    à partir de leur désignation textuelle.

    Rendez-vous sur la page **Prédiction** pour soumettre un produit.
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Catégories", "27", help="Nombre de catégories prdtypecode")
    col2.metric("Modèle", "TF-IDF + classifieur",
                help="Pipeline scikit-learn tracké via MLflow")
    col3.metric("Top-k", "1 à 10",
                help="Nombre de catégories retournées par /predict")

    st.divider()
    st.subheader("Comment ça marche ?")
    c1, c2, c3 = st.columns(3)
    c1.info("**1 — Saisir**\n\nEntrez la désignation du produit (min. 10 caractères)")
    c2.info("**2 — Prédire**\n\nLe modèle analyse le texte et retourne les catégories probables")
    c3.info("**3 — Résultat**\n\nCode catégorie + nom + probabilité associée")

    st.divider()
    st.page_link("pages/1_predict.py",
                 label="🔍 Aller à la prédiction", use_container_width=True)
    st.stop()

# ── Vue ADMIN ─────────────────────────────────────────────────────────────────

# ── Badges liens externes ─────────────────────────────────────────────────────
DAGSHUB_USER = os.getenv("DAGSHUB_USER", "")
DAGSHUB_REPO = os.getenv("DAGSHUB_REPO", "")
GITHUB_REPO = os.getenv("GITHUB_REPO", "")
GITHUB_USER = os.getenv("GITHUB_USER", "")
AIRFLOW_URL = os.getenv("AIRFLOW_URL", "http://localhost:8080")
GRAFANA_URL = os.getenv("GRAFANA_URL", "http://localhost:3000")

dagshub_url = f"https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}"
mlflow_url = f"https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}.mlflow"
github_url = f"https://github.com/{GITHUB_USER}/{GITHUB_REPO}"


st.subheader("Liens externes")
st.markdown(f"""
[![DagsHub](https://img.shields.io/badge/DagsHub-Data%20%26%20Models-1e6b4a?style=for-the-badge&logo=github)]({dagshub_url})
[![MLflow](https://img.shields.io/badge/MLflow-Experiments-0194E2?style=for-the-badge&logo=mlflow)]({mlflow_url})
[![GitHub](https://img.shields.io/badge/GitHub-Source-181717?style=for-the-badge&logo=github)]({github_url})
[![Airflow](https://img.shields.io/badge/Airflow-DAGs-017CEE?style=for-the-badge&logo=apacheairflow&logoColor=white)]({AIRFLOW_URL})
[![Grafana](https://img.shields.io/badge/Grafana-Dashboards-F46800?style=for-the-badge&logo=grafana&logoColor=white)]({GRAFANA_URL})
""", unsafe_allow_html=True)


st.divider()

# ── Architecture ASCII ────────────────────────────────────────────────────────
st.subheader("Architecture du système")

st.code("""
.        ┌─────────────────────────┐
         │       Streamlit UI      │
         └────────────┬────────────┘
                      │ HTTP / JWT
         ┌────────────▼────────────────┐
         │       Gateway FastAPI       │
         │    Auth · routing · proxy   │
         └──┬────┬─────────────┬─────┬─┘
            │    │             │     │
     ┌──────▼─┐ ┌▼─────┐ ┌─────▼──┐ ┌▼─────────┐
     │ingest  │ │train │ │predict │ │ monitor  │
     │CSV·DVC │ │TF-IDF│ │top-k   │ │Evidently │
     └─────┬──┘ └──┬───┘ └───┬────┘ └────┬─────┘
           │       │         │           │
         ┌─▼───────▼─────────▼───────────▼────┐
         │        MLflow · DagsHub · DVC      │
         └────────────────────────────────────┘

  ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐
      Airflow (DAGs · CeleryExecutor)
      orchestration ingest + train
  └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘
""", language=None)

st.divider()

# ── Workflow ──────────────────────────────────────────────────────────────────
st.subheader("Workflow MLOps")
c1, c2, c3, c4 = st.columns(4)
c1.info("**1 — Init / Ingest**\n\nChargement du dataset Rakuten, validation et versioning DVC")
c2.info("**2 — Train**\n\nEntraînement TF-IDF + classifieur, tracking MLflow, push DagsHub")
c3.info("**3 — Predict**\n\nChargement du meilleur modèle MLflow, inférence top-k catégories")
c4.info("**4 — Monitor**\n\nDrift Evidently entre référence et batch courant, métriques loggées")

st.divider()

# ── Accès rapide ──────────────────────────────────────────────────────────────
st.subheader("Accès rapide")
col1, col2, col3, col4 = st.columns(4)
col1.page_link("pages/1_predict.py",    label="🔍 Prédiction",
               use_container_width=True)
col2.page_link("pages/2_pipeline.py",   label="⚙️ Pipeline",
               use_container_width=True)
col3.page_link("pages/3_monitoring.py", label="📊 Monitoring",
               use_container_width=True)
col4.page_link("pages/4_presentation.py", label="🎤 Présentation",
               use_container_width=True)
