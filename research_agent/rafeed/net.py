"""Polite HTTP client: per-host minimum interval, retries with backoff, Retry-After.

Every scholarly API we use publishes a rate rule (arXiv: one request every 3 s on a
single connection; Semantic Scholar: 1 rps with a key; dblp: back off on 429 and wait
1-2 s between requests). The intervals live in config.toml [http.min_interval]."""
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from urllib.parse import urlsplit

RETRY_CODES = (429, 500, 502, 503, 504)


class Http:
    def __init__(self, contact="", min_interval=None, host_headers=None, retries=4,
                 timeout=60, sleep=time.sleep, clock=time.monotonic):
        self.user_agent = f"rafeed/0.1 (mailto:{contact})" if contact else "rafeed/0.1"
        self.min_interval = dict(min_interval or {})
        self.host_headers = dict(host_headers or {})
        self.retries, self.timeout = retries, timeout
        self._sleep, self._clock = sleep, clock
        self._last = {}

    def _throttle(self, host):
        gap = self.min_interval.get(host, self.min_interval.get("*", 1.0))
        last = self._last.get(host)
        if last is not None:
            wait = last + gap - self._clock()
            if wait > 0:
                self._sleep(wait)
        self._last[host] = self._clock()

    def get(self, url, params=None, headers=None, data=None):
        if params:
            url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params, doseq=True)
        host = urlsplit(url).hostname or ""
        hdrs = {"User-Agent": self.user_agent, **self.host_headers.get(host, {}), **(headers or {})}
        for attempt in range(self.retries + 1):
            self._throttle(host)
            req = urllib.request.Request(url, headers=hdrs, data=data)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return resp.read()
            except urllib.error.HTTPError as e:
                if e.code in RETRY_CODES and attempt < self.retries:
                    ra = e.headers.get("Retry-After") if e.headers else None
                    delay = float(ra) if ra and ra.strip().isdigit() else 2.0 ** (attempt + 1)
                    self._sleep(min(delay, 120.0))
                    continue
                raise
            except (urllib.error.URLError, TimeoutError):
                if attempt < self.retries:
                    self._sleep(2.0 ** (attempt + 1))
                    continue
                raise
        raise RuntimeError("unreachable")

    def get_json(self, url, params=None, headers=None, body=None):
        data = json.dumps(body).encode() if body is not None else None
        extra = {"Content-Type": "application/json"} if body is not None else {}
        return json.loads(self.get(url, params, {**extra, **(headers or {})}, data))


class FakeHttp:
    """Offline stand-in for tests and dry runs: maps URL substrings to bytes."""

    def __init__(self, routes):
        self.routes = routes  # list of (substring, bytes)
        self.calls = []

    def get(self, url, params=None, headers=None, data=None):
        if params:
            url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params, doseq=True)
        self.calls.append(url)
        for key, payload in self.routes:
            if key in url:
                return payload
        raise urllib.error.HTTPError(url, 404, "no fixture", None, None)

    def get_json(self, url, params=None, headers=None, body=None):
        return json.loads(self.get(url, params, headers))
