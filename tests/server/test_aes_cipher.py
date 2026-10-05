"""Tests for the AES-256-GCM credential-store cipher."""

from __future__ import annotations

import base64
import secrets

import pytest

from omnigent.stores.credential_store.aes_cipher import (
    CREDENTIAL_AES_KEY_ENV_VAR,
    AesSecretCipher,
    build_aes_secret_cipher,
)
from omnigent.stores.credential_store.secret_cipher import (
    CREDENTIAL_CIPHER_ENV_VAR,
    CREDENTIAL_KMS_KEY_ENV_VAR,
    build_secret_cipher,
)

CTX_ALICE = {"workspace_id": "0", "user_id": "alice", "provider": "databricks", "account_id": ""}
CTX_BOB = {**CTX_ALICE, "user_id": "bob"}


@pytest.fixture
def cipher() -> AesSecretCipher:
    return AesSecretCipher(secrets.token_bytes(32))


def test_round_trip_is_bound_to_the_row_identity(cipher: AesSecretCipher) -> None:
    blob = cipher.encrypt("refresh-token", context=CTX_ALICE)

    assert blob.startswith("aes1:") and "refresh-token" not in blob
    assert cipher.decrypt(blob, context=CTX_ALICE) == "refresh-token"
    assert cipher.decrypt(blob, context=CTX_BOB) is None
    # Empty values are dropped, so an absent account binds like account_id="".
    assert cipher.decrypt(blob, context={k: v for k, v in CTX_ALICE.items() if v}) == (
        "refresh-token"
    )


def test_nonce_is_fresh_per_encryption(cipher: AesSecretCipher) -> None:
    assert cipher.encrypt("x", context=CTX_ALICE) != cipher.encrypt("x", context=CTX_ALICE)


@pytest.mark.parametrize("blob", ["", "kms:abc", "aes1:", "aes1:!!!", "aes1:AAAA"])
def test_foreign_or_corrupt_blobs_decrypt_to_none(cipher: AesSecretCipher, blob: str) -> None:
    assert cipher.decrypt(blob, context=CTX_ALICE) is None


def test_another_key_cannot_decrypt(cipher: AesSecretCipher) -> None:
    blob = cipher.encrypt("x", context=CTX_ALICE)
    assert AesSecretCipher(secrets.token_bytes(32)).decrypt(blob, context=CTX_ALICE) is None


def test_builder_reads_base64_or_base64url_and_rejects_bad_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(CREDENTIAL_AES_KEY_ENV_VAR, raising=False)
    assert build_aes_secret_cipher() is None

    key = secrets.token_bytes(32)
    for encoded in (
        base64.b64encode(key).decode(),
        base64.urlsafe_b64encode(key).decode().rstrip("="),
    ):
        monkeypatch.setenv(CREDENTIAL_AES_KEY_ENV_VAR, encoded)
        built = build_aes_secret_cipher()
        assert built is not None
        assert (
            AesSecretCipher(key).decrypt(built.encrypt("x", context=CTX_ALICE), context=CTX_ALICE)
            == "x"
        )

    monkeypatch.setenv(CREDENTIAL_AES_KEY_ENV_VAR, base64.b64encode(b"short").decode())
    with pytest.raises(ValueError, match="32 bytes"):
        build_aes_secret_cipher()


def test_build_secret_cipher_selects_aes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CREDENTIAL_KMS_KEY_ENV_VAR, raising=False)
    monkeypatch.delenv("OMNIGENT_CREDENTIAL_VAULT_KEY", raising=False)
    monkeypatch.delenv(CREDENTIAL_CIPHER_ENV_VAR, raising=False)
    monkeypatch.setenv(
        CREDENTIAL_AES_KEY_ENV_VAR, base64.b64encode(secrets.token_bytes(32)).decode()
    )

    assert isinstance(build_secret_cipher(), AesSecretCipher)
    monkeypatch.setenv(CREDENTIAL_CIPHER_ENV_VAR, "aes")
    assert isinstance(build_secret_cipher(), AesSecretCipher)
    monkeypatch.setenv(CREDENTIAL_KMS_KEY_ENV_VAR, "alias/x")
    monkeypatch.delenv(CREDENTIAL_CIPHER_ENV_VAR)
    with pytest.raises(ValueError, match="multiple credential backends"):
        build_secret_cipher()
