import streamlit as st
from components.auth import is_authenticated, is_admin
from components import api_client
import httpx

if not is_authenticated() or not is_admin():
    st.error("Accès réservé aux administrateurs.")
    st.stop()

st.title("Pipeline")
st.caption("Déclencher et surveiller les étapes du pipeline MLOps")

# ── Init ──────────────────────────────────────────────────────────────────────
st.subheader("Initialisation des données")
st.caption("Télécharge et prépare le dataset Rakuten depuis la source.")
force = st.checkbox("Forcer le re-téléchargement", value=False)
if st.button("Lancer init", use_container_width=True):
    try:
        with st.spinner("Init en cours..."):
            resp = api_client.post("/init", json=None, timeout=600.0, params={"force": force})
        if resp.status_code == 200:
            st.success("Init terminé.")
            st.json(resp.json())
        else:
            st.error(f"Erreur {resp.status_code} : {resp.text}")
    except httpx.TimeoutException:
        st.error("Timeout — l'init prend trop de temps.")

st.divider()

# ── Ingest ────────────────────────────────────────────────────────────────────
st.subheader("Ingestion des données")
st.caption("Uploader un CSV pour ingestion dans le pipeline.")
uploaded_file = st.file_uploader("Fichier CSV", type=["csv"])
if st.button("Lancer ingest", use_container_width=True, disabled=uploaded_file is None):
    try:
        with st.spinner("Ingest en cours..."):
            resp = api_client.post(
                "/ingest",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")},
            )
        if resp.status_code == 200:
            st.success("Ingest terminé.")
            st.json(resp.json())
        else:
            st.error(f"Erreur {resp.status_code} : {resp.text}")
    except httpx.TimeoutException:
        st.error("Timeout — l'ingest prend trop de temps.")

st.divider()

# ── Train ─────────────────────────────────────────────────────────────────────
st.subheader("Entraînement du modèle")
st.caption("Lance le training puis recharge automatiquement le modèle.")
if st.button("Lancer l'entraînement", use_container_width=True):
    try:
        with st.spinner("Entraînement en cours — peut prendre plusieurs minutes..."):
            resp = api_client.post("/train", timeout=3600.0)
        if resp.status_code == 200:
            result = resp.json()
            st.success(f"Pipeline terminé en {result.get('total_duration_seconds', 0):.0f}s")
            
            col1, col2 = st.columns(2)
            train = result.get("training", {})
            reload = result.get("model_reload", {})
            col1.metric("Training", train.get("status", "—"), f"{train.get('duration_seconds', 0):.0f}s")
            col2.metric("Reload modèle", reload.get("status", "—"), f"{reload.get('duration_seconds', 0):.0f}s")
            
            if result.get("ready_for_predictions"):
                st.success("Modèle prêt pour les prédictions.")
            else:
                st.warning("Le reload a échoué — vérifier les logs.")
            
            with st.expander("Détails complets"):
                st.json(result)
        else:
            st.error(f"Erreur {resp.status_code} : {resp.text}")
    except httpx.TimeoutException:
        st.error("Timeout dépassé — vérifier les logs Airflow.")

st.divider()

# ── Drift ─────────────────────────────────────────────────────────────────────
st.subheader("Monitoring — Drift")
st.caption("Compare le dataset de référence avec le dernier batch ingéré.")

col1, col2 = st.columns(2)

with col1:
    if st.button("Générer rapport drift", use_container_width=True):
        try:
            with st.spinner("Analyse Evidently en cours..."):
                resp = api_client.post("/drift", timeout=300.0)
            if resp.status_code == 200:
                st.success("Rapport généré.")
                st.json(resp.json())
            else:
                st.error(f"Erreur {resp.status_code} : {resp.text}")
        except httpx.TimeoutException:
            st.error("Timeout — analyse trop longue.")

with col2:
    if st.button("Afficher rapport HTML", use_container_width=True):
        resp = api_client.get("/drift/report")
        if resp.status_code == 200:
            st.components.v1.html(resp.text, height=600, scrolling=True)
        else:
            st.warning("Aucun rapport disponible — lancer d'abord l'analyse.")

st.divider()

# ── Health ────────────────────────────────────────────────────────────────────
st.subheader("Statut de la gateway")
if st.button("Rafraîchir", use_container_width=False):
    resp = api_client.get("/health")
    if resp.status_code == 200:
        st.success("Gateway opérationnelle.")
        st.json(resp.json())
    else:
        st.warning("Gateway inaccessible.")

# ── Airflow ───────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Airflow")
st.markdown("[Ouvrir Airflow →](http://localhost:8080)", unsafe_allow_html=True)
st.caption("airflow / airflow — interface CeleryExecutor")