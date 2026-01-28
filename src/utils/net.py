from __future__ import annotations

import socket
from urllib.parse import urlparse


def tcp_check(host: str, port: int, timeout_s: float = 0.5) -> tuple[bool, str]:
    """Best-effort TCP reachability check."""
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True, ""
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def parse_host_port(uri: str, default_port: int) -> tuple[str, int]:
    """
    Accepts:
    - bolt://host:7687
    - http://host:19530
    - ws://host:10095
    - host:port
    - [::1]:1234
    """
    uri = (uri or "").strip()
    if not uri:
        return "localhost", default_port

    if "://" in uri:
        u = urlparse(uri)
        host = u.hostname or "localhost"
        port = u.port or default_port
        return host, port

    # Handle plain host:port (without scheme). Be conservative with IPv6-like inputs.
    if ":" in uri and not uri.startswith("["):
        # Likely IPv6 without brackets -> treat as host-only.
        if uri.count(":") > 1:
            return uri, default_port

        host, port_s = uri.rsplit(":", 1)
        try:
            return host, int(port_s)
        except Exception:  # noqa: BLE001
            return uri, default_port

    # Bracketed IPv6: [::1]:1234
    if uri.startswith("[") and "]" in uri:
        host_part, _, rest = uri.partition("]")
        host = host_part[1:]
        if rest.startswith(":"):
            try:
                return host, int(rest[1:])
            except Exception:  # noqa: BLE001
                return host, default_port
        return host, default_port

    return uri, default_port
