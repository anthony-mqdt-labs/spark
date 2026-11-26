"""Port selection for the supervised server.

Kept separate from the supervisor so the launch loop does not own socket policy.
The supervisor re-exports these names for existing importers.
"""

from __future__ import annotations

import socket

from ..errors import PortInUseError


def _port_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def reserve_port(host: str, port: int) -> socket.socket:
    """Bind + listen and return the OPEN socket, so nobody can take the port.

    The caller owns the returned socket and must close it. This is the handoff
    seam for children spark launches itself (the mlx_lm filter-proxy shim): the
    supervisor holds the port from selection until ``Popen`` passes the fd down,
    and the child adopts it without a second bind.

    Backends that bind their own port still use ``_port_free``/``_pick_port``:
    they probe then release, then the runtime binds much later. That leaves a
    small residual window spark cannot close without runtime cooperation.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind((host, port))
        s.listen(1)
    except BaseException:
        s.close()
        raise
    return s


def _pick_port(host: str, preferred: int, lo: int, hi: int) -> int:
    if _port_free(host, preferred):
        return preferred
    for p in range(lo, hi + 1):
        if _port_free(host, p):
            return p
    raise PortInUseError(
        f"No free port in range {lo}-{hi} (preferred {preferred} busy).",
        remediation=[
            "Free a port or widen general.port_range in your config.",
            f"Find the holder: lsof -nP -iTCP:{preferred} -sTCP:LISTEN",
        ],
        context={"preferred": preferred, "range": [lo, hi]},
    )
