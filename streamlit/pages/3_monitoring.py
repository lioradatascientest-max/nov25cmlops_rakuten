import streamlit as st
import os
from components.auth import is_authenticated, is_admin
from components import api_client
import httpx

if not is_authenticated() or not is_admin():
    st.error("Accès réservé aux administrateurs.")
    st.stop()

DAGSHUB_USER = os.getenv("DAGSHUB_USER", "")
DAGSHUB_REPO = os.getenv("DAGSHUB_REPO", "")

st.title("Monitoring")

# ── Drift Evidently ───────────────────────────────────────────────────────────
st.subheader("Drift — Evidently")

col1, col2 = st.columns(2)
with col1:
    if st.button("Générer rapport drift", use_container_width=True):
        try:
            with st.spinner("Calcul du drift..."):
                resp = api_client.post("/drift", timeout=300.0)
            if resp.status_code == 200:
                st.success("Rapport généré.")
                st.json(resp.json())
            else:
                st.error(f"Erreur {resp.status_code} : {resp.text}")
        except httpx.TimeoutException:
            st.error("Timeout — analyse trop longue.")

with col2:
    if st.button("Voir rapport HTML", use_container_width=True):
        try:
            resp = api_client.get("/drift/report")
            if resp.status_code == 200:
                st.components.v1.html(resp.text, height=600, scrolling=True)
            else:
                st.warning("Rapport non disponible — générez-le d'abord.")
        except httpx.ConnectError:
            st.error("Gateway inaccessible.")

st.divider()

# ── Modèle en production ──────────────────────────────────────────────────────
st.subheader("Modèle en production")

if st.button("Charger les infos modèle", use_container_width=False):
    try:
        resp = api_client.get("/info")
        if resp.status_code == 200:
            info = resp.json()
            st.json(info)
        else:
            st.warning(f"Erreur {resp.status_code} : {resp.text}")
    except httpx.ConnectError:
        st.error("Gateway inaccessible.")

st.divider()

# ── MLflow / DagsHub ──────────────────────────────────────────────────────────
st.subheader("MLflow — DagsHub")

if DAGSHUB_USER and DAGSHUB_REPO:
    mlflow_url = f"https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}.mlflow"
    st.markdown(f"[Ouvrir MLflow sur DagsHub →]({mlflow_url})")
    st.caption("DagsHub bloque les iframes — accès via lien externe uniquement.")
else:
    st.warning("DAGSHUB_USER ou DAGSHUB_REPO non configuré dans les variables d'environnement.")

st.divider()

# ── Grafana ───────────────────────────────────────────────────────────────────
st.subheader("Grafana")
st.markdown("[Ouvrir Grafana →](http://localhost:3000)", unsafe_allow_html=True)
st.caption("admin / admin — dashboards Prometheus / Nginx")
grafana_url = "http://localhost:3000/d/rakuten?orgId=1&kiosk=tv"
st.components.v1.iframe(grafana_url, height=500, scrolling=True)