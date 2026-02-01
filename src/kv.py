# kv.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient


@dataclass(frozen=True)
class Settings:
    # Example settings you might store in Key Vault
    sql_conn_str: Optional[str] = None
    openai_api_key: Optional[str] = None
    some_third_party_key: Optional[str] = None


def _kv_uri_from_env() -> str:
    """
    Requires KEYVAULT_NAME in App Service configuration.
    Example: KEYVAULT_NAME = troywfm-dev-kv
    """
    name = os.getenv("KEYVAULT_NAME", "").strip()
    if not name:
        raise RuntimeError(
            "Missing KEYVAULT_NAME environment variable. "
            "Set it in App Service Configuration (or your local env)."
        )
    return f"https://{name}.vault.azure.net/"


def _secret(client: SecretClient, name: str) -> Optional[str]:
    """
    Return a secret value or None if it doesn't exist or is inaccessible.
    Keep this tolerant so the app can still run with partial configuration.
    """
    try:
        return client.get_secret(name).value
    except Exception:
        return None


def load_settings_from_key_vault(
    *,
    kv_uri: Optional[str] = None,
    # Map your Key Vault secret names here
    sql_secret_name: str = "sql-conn-str",
    openai_secret_name: str = "openai-api-key",
    third_party_secret_name: str = "third-party-api-key",
) -> Settings:
    """
    Uses Managed Identity in Azure (DefaultAzureCredential) and also works locally
    via Azure CLI login, VS Code credentials, etc.
    """
    uri = kv_uri or _kv_uri_from_env()

    # DefaultAzureCredential tries (in order): Managed Identity, VS Code, Azure CLI, etc.
    credential = DefaultAzureCredential(
        exclude_interactive_browser_credential=True  # Streamlit shouldn't pop browsers in prod
    )
    client = SecretClient(vault_url=uri, credential=credential)

    return Settings(
        sql_conn_str=_secret(client, sql_secret_name),
        openai_api_key=_secret(client, openai_secret_name),
        some_third_party_key=_secret(client, third_party_secret_name),
    )
