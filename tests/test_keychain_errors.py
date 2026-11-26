"""Keychain status mapping and overwrite flow with a fake Security framework.

Nothing here touches the real macOS Keychain: ``store._lib`` is replaced with a
small in-memory stand-in before any operation. Only the non-secret names index
lives on disk, under ``tmp_path``.
"""

from __future__ import annotations

import ctypes
from ctypes import POINTER, c_uint32, c_void_p, cast, create_string_buffer

import pytest

from spark.errors import SecretBackendError, SecretError
from spark.secrets.keychain import KeychainStore

_NOT_FOUND = -25300
_DUPLICATE = -25299
_AUTH_FAILED = -25293
_INTERACTION_NOT_ALLOWED = -25308


class FakeLib:
    """Minimal Security-framework stand-in (add / find / delete / free)."""

    def __init__(self, statuses: dict[str, int] | None = None) -> None:
        self.items: dict[str, bytes] = {}
        self.calls: list[tuple[str, str]] = []
        self.statuses = dict(statuses or {})
        self._buffers: list[ctypes.Array] = []
        self._last_find: str | None = None

    def _status(self, op: str) -> int:
        return self.statuses.get(op, 0)

    def SecKeychainAddGenericPassword(self, _kc, _sl, _service, _al, acct, _dl, data, _item):
        name = acct.decode()
        self.calls.append(("add", name))
        if self._status("add"):
            return self._status("add")
        if name in self.items:
            return _DUPLICATE
        self.items[name] = bytes(data)
        return 0

    def SecKeychainFindGenericPassword(
        self, _kc, _sl, _service, _al, acct, length_ptr, data_ptr, item_ref
    ):
        name = acct.decode()
        self.calls.append(("find", name))
        if self._status("find"):
            return self._status("find")
        if name not in self.items:
            return _NOT_FOUND
        self._last_find = name
        cast(item_ref, POINTER(c_void_p))[0] = 1
        if length_ptr is not None and data_ptr is not None:
            raw = self.items[name]
            buf = create_string_buffer(raw)
            self._buffers.append(buf)  # keep the backing bytes alive
            cast(length_ptr, POINTER(c_uint32))[0] = len(raw)
            cast(data_ptr, POINTER(c_void_p))[0] = cast(buf, c_void_p).value
        return 0

    def SecKeychainItemFreeContent(self, _attrs, _data_ptr):
        self.calls.append(("free", ""))
        return 0

    def SecKeychainItemDelete(self, _item):
        self.calls.append(("delete", self._last_find or ""))
        if self._status("delete"):
            return self._status("delete")
        self.items.pop(self._last_find or "", None)
        return 0


def _store(tmp_path, statuses=None):
    store = KeychainStore(index_path=tmp_path / "names-index.json")
    store._lib = FakeLib(statuses)
    return store, store._lib


@pytest.mark.parametrize(
    "status, marker, exact_type",
    [
        (_AUTH_FAILED, "access denied", SecretError),
        (_INTERACTION_NOT_ALLOWED, "locked", SecretError),
        (-1, "OSStatus", SecretBackendError),
    ],
)
def test_status_mapping_is_typed_with_remediation_and_context(tmp_path, status, marker, exact_type):
    store, _ = _store(tmp_path)
    with pytest.raises(SecretError) as exc:
        store._raise_status(status, "hf_token", "get")
    assert type(exc.value) is exact_type
    assert marker in str(exc.value)
    assert "hf_token" in str(exc.value)
    assert exc.value.remediation
    assert exc.value.context == {"op": "get", "status": status}


@pytest.mark.parametrize(
    "status, method, op",
    [
        (_AUTH_FAILED, "get", "get"),
        (_INTERACTION_NOT_ALLOWED, "exists", "exists"),
    ],
)
def test_operation_reports_the_failing_op(tmp_path, status, method, op):
    store, _ = _store(tmp_path, statuses={"find": status})
    with pytest.raises(SecretError) as exc:
        getattr(store, method)("hf_token")
    assert exc.value.context["op"] == op
    assert exc.value.context["status"] == status


def test_generic_failure_during_add_leaves_index_untouched(tmp_path):
    store, _ = _store(tmp_path, statuses={"add": -1})
    with pytest.raises(SecretBackendError) as exc:
        store.set("hf_token", "some-value")
    assert exc.value.context == {"op": "set", "status": -1}
    assert exc.value.remediation
    assert store._index_read() == []


def test_duplicate_set_deletes_then_readds(tmp_path):
    store, lib = _store(tmp_path)
    store.set("hf_token", "first-value")
    store.set("hf_token", "second-value")

    assert lib.calls == [
        ("add", "hf_token"),
        ("add", "hf_token"),  # duplicate detected
        ("find", "hf_token"),  # attribute-only lookup before delete
        ("delete", "hf_token"),
        ("add", "hf_token"),  # re-add with the new value
    ]
    assert lib.items["hf_token"] == b"second-value"
    assert store.get("hf_token") == "second-value"
    assert store._index_read() == ["hf_token"]
