"""HTTP sessions with a real timeout for pyvo.

pyvo's `run_sync(..., timeout=...)` is passed to the service as a query parameter, not to the socket, so a service
that accepts the connection and never answers would block forever. These sessions put a timeout on every request.
"""

import requests


class TimeoutSession(requests.Session):
    def __init__(self, timeout_s: float) -> None:
        super().__init__()
        self.timeout_s = timeout_s

    def request(self, method, url, **kwargs):
        kwargs.setdefault("timeout", self.timeout_s)
        return super().request(method, url, **kwargs)
