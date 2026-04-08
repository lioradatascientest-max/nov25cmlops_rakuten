import streamlit as st
from components.auth import is_authenticated
from components import api_client
import httpx

if not is_authenticated():
    st.warning("Veuillez vous connecter.")
    st.stop()

st.title("Prédiction produit")
st.caption("Soumettre un produit Rakuten pour classification automatique")

with st.form("predict_form"):
    designation = st.text_input(
        "Désignation *",
        placeholder="ex: Lot de 4 couverts inox...",
        help="Minimum 10 caractères"
    )
    top_k = st.slider("Nombre de catégories à afficher", min_value=1, max_value=10, value=1)
    submitted = st.form_submit_button("Prédire", use_container_width=True)

if submitted:
    if len(designation) < 10:
        st.error("La désignation doit faire au moins 10 caractères.")
    else:
        try:
            with st.spinner("Prédiction en cours..."):
                resp = api_client.post(
                    "/predict",
                    json={"designation": designation, "top_k": top_k},
                )

            if resp.status_code == 200:
                result = resp.json()
                predictions = result.get("predictions", [])

                if top_k == 1 and predictions:
                    best = predictions[0]
                    col1, col2 = st.columns(2)
                    col1.metric("Catégorie prédite", best.get("category_name") or str(best["prdtypecode"]))
                    col2.metric("Probabilité", f"{best['proba']:.1%}")
                else:
                    st.subheader("Top catégories")
                    for i, pred in enumerate(predictions, 1):
                        name = pred.get("category_name") or f"Code {pred['prdtypecode']}"
                        st.progress(pred["proba"], text=f"{i}. {name} — {pred['proba']:.1%}")

            elif resp.status_code == 401:
                st.error("Session expirée, veuillez vous reconnecter.")
                st.session_state.clear()
            elif resp.status_code == 422:
                st.error("Données invalides.")
                st.json(resp.json())  # à retirer en prod
            else:
                st.error(f"Erreur {resp.status_code} : {resp.text}")

        except httpx.TimeoutException:
            st.error("Le service de prédiction ne répond pas (timeout).")
        except httpx.ConnectError:
            st.error("Impossible de joindre la gateway.")