import httpx
import streamlit as st
import os
from typing import Any

GATEWAY_URL = os.getenv("GATEWAY_URL", "http://gateway:8000")


def _headers() -> dict:
    token = st.session_state.get("token", "")
    return {"Authorization": f"Bearer {token}"}

# Api client pour la gateway, avec le token, timeout et files et params.
def get(path: str, params: Any = None, timeout: float = 10.0, **kwargs) -> httpx.Response:
    return httpx.get(
        f"{GATEWAY_URL}{path}",
        headers=_headers(),
        params=params,
        timeout=timeout,
        **kwargs,
    )


def post(path: str, json: Any = None, data: Any = None, files: Any = None,
         params: Any = None, timeout: float = 30.0, **kwargs) -> httpx.Response:
    return httpx.post(
        f"{GATEWAY_URL}{path}",
        headers=_headers(),
        json=json,
        data=data,
        files=files,
        params=params,
        timeout=timeout,
        **kwargs,
    )