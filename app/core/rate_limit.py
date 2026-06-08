"""Shared rate limiter (slowapi) used to throttle abuse-prone endpoints —
auth/OTP/email routes — by client IP.

Note: the default storage is in-memory, which is correct for a single worker.
For a multi-worker / multi-instance deployment, point slowapi at Redis via
`Limiter(key_func=..., storage_uri="redis://...")` so limits are shared.
Behind a reverse proxy, ensure the real client IP reaches the app (e.g. via
X-Forwarded-For handling / trusted-proxy config) so limits key off the user,
not the proxy.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
