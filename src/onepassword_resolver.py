"""Helpers for resolving database credentials from 1Password."""

from __future__ import annotations

import asyncio
import os
import threading
from typing import Any, Optional

try:
    from onepassword import Client
except ImportError:  # pragma: no cover - handled gracefully at runtime
    Client = None


class OnePasswordResolver:
    """Resolve connection secrets from 1Password using a service account."""

    _instance: Optional["OnePasswordResolver"] = None

    def __init__(self):
        self._client = None
        self._lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> "OnePasswordResolver":
        """Get the shared resolver instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def is_configured(self) -> bool:
        """Return whether 1Password resolution can be used."""
        return bool(Client is not None and os.getenv("OP_SERVICE_ACCOUNT_TOKEN"))

    def resolve_connection_values(self, connection: dict[str, Any]) -> dict[str, str]:
        """Resolve credential values for a connection from stored 1Password references."""
        resolved: dict[str, str] = {}

        connection_string_ref = connection.get("connectionStringSecretRef")
        if connection_string_ref:
            resolved_connection_string = self.resolve_secret_reference(connection_string_ref)
            if resolved_connection_string:
                resolved["connectionString"] = resolved_connection_string

        password_ref = connection.get("passwordSecretRef")
        if password_ref:
            resolved_password = self.resolve_secret_reference(password_ref)
            if resolved_password:
                resolved["password"] = resolved_password

        needs_item_lookup = not resolved.get("connectionString") or not resolved.get("password")
        if not needs_item_lookup:
            return resolved

        vault_id = connection.get("credentialVaultId")
        item_id = connection.get("credentialItemId")
        if not vault_id or not item_id:
            return resolved

        item_fields = self.get_item_fields(vault_id, item_id)

        if not resolved.get("connectionString"):
            connection_string = item_fields.get("connectionString") or item_fields.get("database_details.connectionString")
            if connection_string:
                resolved["connectionString"] = connection_string

        if not resolved.get("password"):
            password = item_fields.get("password")
            if password:
                resolved["password"] = password

        if not connection.get("username"):
            username = item_fields.get("username")
            if username:
                resolved["username"] = username

        return resolved

    def resolve_secret_reference(self, secret_reference: str) -> Optional[str]:
        """Resolve a single secret reference."""
        if not self.is_configured():
            return None

        async def _resolve() -> str:
            client = await self._get_client()
            return await client.secrets.resolve(secret_reference)

        return self._run_async(_resolve())

    def get_item_fields(self, vault_id: str, item_id: str) -> dict[str, str]:
        """Fetch an item and return its fields keyed by ID and title."""
        if not self.is_configured():
            return {}

        async def _get_fields() -> dict[str, str]:
            client = await self._get_client()
            item = await client.items.get(vault_id, item_id)
            fields: dict[str, str] = {}

            for field in item.fields or []:
                value = getattr(field, "value", None)
                if not isinstance(value, str) or value == "":
                    continue

                field_id = getattr(field, "id", None)
                field_title = getattr(field, "title", None)
                section_id = getattr(field, "section_id", None)

                if field_id:
                    fields[field_id] = value
                if field_title:
                    fields[field_title] = value
                if section_id and field_id:
                    fields[f"{section_id}.{field_id}"] = value
                if section_id and field_title:
                    fields[f"{section_id}.{field_title}"] = value

            return fields

        return self._run_async(_get_fields())

    async def _get_client(self):
        if Client is None:
            raise RuntimeError("onepassword-sdk is not installed")

        if self._client is not None:
            return self._client

        with self._lock:
            if self._client is None:
                token = os.getenv("OP_SERVICE_ACCOUNT_TOKEN")
                if not token:
                    raise RuntimeError("OP_SERVICE_ACCOUNT_TOKEN is not set")

                self._client = await Client.authenticate(
                    auth=token,
                    integration_name="db-agent",
                    integration_version="v0.1.0",
                )

        return self._client

    def _run_async(self, coroutine):
        """Run an async coroutine from sync code, even if an event loop is active."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coroutine)

        result: dict[str, Any] = {}
        error: dict[str, BaseException] = {}

        def _runner() -> None:
            try:
                result["value"] = asyncio.run(coroutine)
            except BaseException as exc:  # pragma: no cover - passthrough for runtime issues
                error["value"] = exc

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        thread.join()

        if "value" in error:
            raise error["value"]

        return result.get("value")


def get_onepassword_resolver() -> OnePasswordResolver:
    """Get the shared 1Password resolver."""
    return OnePasswordResolver.get_instance()