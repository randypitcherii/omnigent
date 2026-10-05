"""AES-256-GCM :class:`SecretCipher` keyed by a deployment-held secret.

For platforms that offer no KMS or Vault but do offer a secret manager that can
inject one value into the server's environment (e.g. a Databricks App with a
secret-scope resource). The 32-byte key comes from
``OMNIGENT_CREDENTIAL_AES_KEY`` (base64 or base64url, padding optional); unset ⇒
this backend is not configured.

Each ciphertext is bound to its row's identity: the context
(``workspace``/``user``/``provider``/``account``) is the GCM *associated data*,
so a ciphertext presented under any other identity fails authentication and
:meth:`AesSecretCipher.decrypt` returns ``None`` (⇒ reconnect), never a wrong
plaintext. Unlike KMS, the key lives in the server process: this protects the
secrets at rest in the database, not against a compromised server.

Format: ``aes1:`` + base64url(12-byte nonce ‖ ciphertext ‖ 16-byte tag).
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import secrets

from omnigent.stores.credential_store.secret_cipher import SecretContext

#: Base64 (or base64url) encoding of the 32-byte AES-256 key. Generate one with
#: ``openssl rand -base64 32``.
CREDENTIAL_AES_KEY_ENV_VAR = "OMNIGENT_CREDENTIAL_AES_KEY"

_PREFIX = "aes1:"
_NONCE_BYTES = 12


def _decode_key(raw: str) -> bytes:
    """Decode a base64/base64url key and require exactly 32 bytes."""
    value = raw.strip()
    padded = value + "=" * (-len(value) % 4)
    try:
        key = base64.urlsafe_b64decode(padded.replace("+", "-").replace("/", "_"))
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"{CREDENTIAL_AES_KEY_ENV_VAR} is not valid base64") from exc
    if len(key) != 32:
        raise ValueError(
            f"{CREDENTIAL_AES_KEY_ENV_VAR} must decode to 32 bytes (AES-256), got {len(key)}"
        )
    return key


def _associated_data(context: SecretContext) -> bytes:
    """Canonical row identity: sorted-key JSON, empty values dropped.

    Empty values are dropped on both encrypt and decrypt (as the KMS and Vault
    ciphers do), so ``account_id=""`` and an absent account bind identically.
    JSON keeps the encoding unambiguous: no two identities share bytes.
    """
    kept = {k: v for k, v in context.items() if v != ""}
    return json.dumps(kept, sort_keys=True, separators=(",", ":")).encode()


class AesSecretCipher:
    """AES-256-GCM cipher bound to each row's identity via associated data.

    :param key: The 32-byte AES-256 key.
    """

    def __init__(self, key: bytes) -> None:
        if len(key) != 32:
            raise ValueError("AesSecretCipher needs a 32-byte key")
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        self._aead = AESGCM(key)

    def encrypt(self, plaintext: str, *, context: SecretContext) -> str:
        """Encrypt *plaintext* bound to *context*; a fresh random nonce per call."""
        nonce = secrets.token_bytes(_NONCE_BYTES)
        sealed = self._aead.encrypt(nonce, plaintext.encode(), _associated_data(context))
        return _PREFIX + base64.urlsafe_b64encode(nonce + sealed).decode().rstrip("=")

    def decrypt(self, ciphertext: str, *, context: SecretContext) -> str | None:
        """Return the plaintext, or ``None`` for a foreign/corrupt/mismatched blob."""
        from cryptography.exceptions import InvalidTag

        if not ciphertext.startswith(_PREFIX):
            return None
        body = ciphertext[len(_PREFIX) :]
        try:
            raw = base64.urlsafe_b64decode(body + "=" * (-len(body) % 4))
        except (binascii.Error, ValueError):
            return None
        if len(raw) <= _NONCE_BYTES:
            return None
        nonce, sealed = raw[:_NONCE_BYTES], raw[_NONCE_BYTES:]
        try:
            plaintext = self._aead.decrypt(nonce, sealed, _associated_data(context))
        except InvalidTag:
            return None
        return plaintext.decode()


def build_aes_secret_cipher() -> AesSecretCipher | None:
    """Build the AES cipher from :data:`CREDENTIAL_AES_KEY_ENV_VAR`, or ``None`` when unset.

    :raises ValueError: The key is set but is not 32 bytes of base64.
    """
    raw = os.environ.get(CREDENTIAL_AES_KEY_ENV_VAR, "").strip()
    if not raw:
        return None
    return AesSecretCipher(_decode_key(raw))
