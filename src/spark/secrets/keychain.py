"""macOS Keychain secret store via the Security framework (ctypes).

Why ctypes and not the ``security`` CLI: ``security add-generic-password -w`` can
only take the value on argv (visible in ``ps``) or from the tty — neither is
acceptable. Calling ``SecKeychainAddGenericPassword`` directly keeps the value in
process memory only: no argv, no temp file, no dependency.

Enumeration (``list_names``) uses a small *non-secret* names index in the data dir
— names like ``hf_token`` are not secrets; values never leave the Keychain.
"""

from __future__ import annotations

import ctypes
import json
import os
import platform
from ctypes import POINTER, byref, c_char_p, c_uint32, c_void_p
from pathlib import Path

from ..errors import SecretBackendError, SecretError, SecretNotFoundError
from .store import SecretStore, validate_name

_SERVICE = "spark"

# OSStatus codes we care about.
_ERR_SEC_SUCCESS = 0
_ERR_SEC_ITEM_NOT_FOUND = -25300
_ERR_SEC_DUPLICATE_ITEM = -25299
_ERR_SEC_AUTH_FAILED = -25293
_ERR_SEC_INTERACTION_NOT_ALLOWED = -25308


def _load_security() -> ctypes.CDLL | None:
    if platform.system() != "Darwin":
        return None
    try:
        lib = ctypes.CDLL(
            "/System/Library/Frameworks/Security.framework/Security"
        )
    except OSError:
        return None

    lib.SecKeychainAddGenericPassword.argtypes = [
        c_void_p, c_uint32, c_char_p, c_uint32, c_char_p,
        c_uint32, c_char_p, POINTER(c_void_p),
    ]
    lib.SecKeychainAddGenericPassword.restype = ctypes.c_int32

    lib.SecKeychainFindGenericPassword.argtypes = [
        c_void_p, c_uint32, c_char_p, c_uint32, c_char_p,
        POINTER(c_uint32), POINTER(c_void_p), POINTER(c_void_p),
    ]
    lib.SecKeychainFindGenericPassword.restype = ctypes.c_int32

    lib.SecKeychainItemFreeContent.argtypes = [c_void_p, c_void_p]
    lib.SecKeychainItemFreeContent.restype = ctypes.c_int32

    lib.SecKeychainItemDelete.argtypes = [c_void_p]
    lib.SecKeychainItemDelete.restype = ctypes.c_int32
    return lib


class KeychainStore(SecretStore):
    backend_name = "macos-keychain"

    def __init__(self, index_path: Path, service: str = _SERVICE) -> None:
        self._lib = _load_security()
        self._service = service.encode("utf-8")
        self._index_path = Path(index_path)

    # -- availability ----------------------------------------------------------
    def is_available(self) -> bool:
        return self._lib is not None

    def _require(self) -> ctypes.CDLL:
        if self._lib is None:
            raise SecretBackendError(
                "macOS Keychain is not available on this host.",
                remediation=[
                    "spark's Keychain backend requires macOS.",
                    "On Linux, configure an alternative secret backend (roadmap).",
                ],
            )
        return self._lib

    # -- status mapping --------------------------------------------------------
    def _raise_status(self, status: int, name: str, op: str) -> None:
        if status == _ERR_SEC_AUTH_FAILED:
            raise SecretError(
                f"Keychain access denied for '{name}'.",
                code="SEC_BACKEND",
                remediation=[
                    "Unlock your login keychain (log out/in, or Keychain Access).",
                    "Approve the access prompt if one appeared.",
                ],
                context={"op": op, "status": status},
            )
        if status == _ERR_SEC_INTERACTION_NOT_ALLOWED:
            raise SecretError(
                f"Keychain locked (no UI) while accessing '{name}'.",
                code="SEC_BACKEND",
                remediation=["Unlock the login keychain, then retry."],
                context={"op": op, "status": status},
            )
        raise SecretBackendError(
            f"Keychain {op} failed for '{name}' (OSStatus {status}).",
            context={"op": op, "status": status},
        )

    # -- core ops --------------------------------------------------------------
    def set(self, name: str, value: str) -> None:
        validate_name(name)
        if value == "":
            raise SecretError(
                "Refusing to store an empty secret value.",
                remediation=["Provide a non-empty value via getpass/stdin."],
            )
        lib = self._require()
        acct = name.encode("utf-8")
        data = value.encode("utf-8")

        item_ref = c_void_p()
        status = lib.SecKeychainAddGenericPassword(
            None,
            len(self._service), self._service,
            len(acct), acct,
            len(data), data,
            byref(item_ref),
        )
        if status == _ERR_SEC_DUPLICATE_ITEM:
            # Overwrite = delete then add (simpler + reliable vs. modify-in-place).
            self._delete_raw(name)
            status = lib.SecKeychainAddGenericPassword(
                None,
                len(self._service), self._service,
                len(acct), acct,
                len(data), data,
                byref(item_ref),
            )
        if status != _ERR_SEC_SUCCESS:
            self._raise_status(status, name, "set")
        self._index_add(name)

    def get(self, name: str) -> str:
        validate_name(name)
        lib = self._require()
        acct = name.encode("utf-8")
        length = c_uint32(0)
        data_ptr = c_void_p()
        item_ref = c_void_p()
        status = lib.SecKeychainFindGenericPassword(
            None,
            len(self._service), self._service,
            len(acct), acct,
            byref(length), byref(data_ptr),
            byref(item_ref),
        )
        if status == _ERR_SEC_ITEM_NOT_FOUND:
            raise SecretNotFoundError(
                f"Secret '{name}' not found.",
                remediation=[f"Store it with: spark secret set {name}"],
                context={"name": name},
            )
        if status != _ERR_SEC_SUCCESS:
            self._raise_status(status, name, "get")
        try:
            raw = ctypes.string_at(data_ptr, length.value)
            value = raw.decode("utf-8", errors="strict")
        finally:
            lib.SecKeychainItemFreeContent(None, data_ptr)
        # Backstop: register so telemetry can never accidentally log it.
        return self._register_for_redaction(value)

    def _delete_raw(self, name: str) -> bool:
        lib = self._require()
        acct = name.encode("utf-8")
        length = c_uint32(0)
        data_ptr = c_void_p()
        item_ref = c_void_p()
        status = lib.SecKeychainFindGenericPassword(
            None,
            len(self._service), self._service,
            len(acct), acct,
            byref(length), byref(data_ptr), byref(item_ref),
        )
        if status == _ERR_SEC_ITEM_NOT_FOUND:
            return False
        if status != _ERR_SEC_SUCCESS:
            self._raise_status(status, name, "delete-find")
        # Free the data buffer we don't need; keep item_ref for deletion.
        lib.SecKeychainItemFreeContent(None, data_ptr)
        del_status = lib.SecKeychainItemDelete(item_ref)
        if del_status != _ERR_SEC_SUCCESS:
            self._raise_status(del_status, name, "delete")
        return True

    def delete(self, name: str) -> None:
        validate_name(name)
        self._delete_raw(name)
        self._index_remove(name)

    def exists(self, name: str) -> bool:
        validate_name(name)
        lib = self._require()
        acct = name.encode("utf-8")
        length = c_uint32(0)
        data_ptr = c_void_p()
        item_ref = c_void_p()
        status = lib.SecKeychainFindGenericPassword(
            None,
            len(self._service), self._service,
            len(acct), acct,
            byref(length), byref(data_ptr), byref(item_ref),
        )
        if status == _ERR_SEC_SUCCESS:
            lib.SecKeychainItemFreeContent(None, data_ptr)
            return True
        if status == _ERR_SEC_ITEM_NOT_FOUND:
            return False
        self._raise_status(status, name, "exists")
        return False  # unreachable

    def list_names(self) -> list[str]:
        """Return registered secret names (non-secret) from the index, keeping only
        those that still resolve in the Keychain."""
        names = self._index_read()
        live = [n for n in names if self.exists(n)]
        if live != names:
            self._index_write(live)
        return sorted(live)

    # -- non-secret names index ------------------------------------------------
    def _index_read(self) -> list[str]:
        try:
            with self._index_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                return [str(x) for x in data]
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        return []

    def _index_write(self, names: list[str]) -> None:
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._index_path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(sorted(set(names)), fh, indent=2)
        os.replace(tmp, self._index_path)
        try:
            os.chmod(self._index_path, 0o600)
        except OSError:
            pass

    def _index_add(self, name: str) -> None:
        names = set(self._index_read())
        names.add(name)
        self._index_write(sorted(names))

    def _index_remove(self, name: str) -> None:
        names = set(self._index_read())
        names.discard(name)
        self._index_write(sorted(names))
