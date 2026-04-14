import streamlit as st
import httpx
import os

GATEWAY_URL = os.getenv("GATEWAY_URL", "http://gateway:8000")

#le décodage est délégué au gateway via GET /me.

def login_form() -> None:
    """Affiche le formulaire de login et gère l'authentification."""
    st.title("Rakuten MLOps")
    st.caption("Connectez-vous pour accéder au dashboard")

    with st.form("login_form"):
        username = st.text_input("Utilisateur")
        password = st.text_input("Mot de passe", type="password")
        submitted = st.form_submit_button("Connexion", use_container_width=True)

    if submitted:
        result = _get_token_and_role(username, password)
        if result:
            token, role = result
            st.session_state["token"] = token
            st.session_state["username"] = username
            st.session_state["role"] = role
            st.rerun()
        else:
            st.error("Identifiants incorrects")


def logout() -> None:
    for key in ("token", "username", "role"):
        st.session_state.pop(key, None)
    st.rerun()


def is_authenticated() -> bool:
    return bool(st.session_state.get("token"))


def is_admin() -> bool:
    return st.session_state.get("role") == "admin"


def get_token() -> str:
    return st.session_state.get("token", "")


# ── Privé ────────────────────────────────────────────────────────────────────

def _get_token_and_role(username: str, password: str) -> tuple[str, str] | None:
    """POST /token puis GET /me → (jwt, role).

    Le rôle est résolu par le gateway (get_current_user de FastAPI),
    """
    try:
        # Obtenir le token
        resp = httpx.post(
            f"{GATEWAY_URL}/token",
            data={"username": username, "password": password},
            timeout=5.0,
        )
        if resp.status_code != 200:
            return None
        token = resp.json().get("access_token")
        if not token:
            return None

        # Résoudre le rôle via /me 
        me = httpx.get(
            f"{GATEWAY_URL}/me",
            headers={"Authorization": f"Bearer {token}"},
            timeout=5.0,
        )
        role = me.json().get("role", "user") if me.status_code == 200 else "user"
        return token, role

    except httpx.RequestError:
        st.error("Gateway inaccessible")
        return None