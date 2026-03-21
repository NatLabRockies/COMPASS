"""Shared URL utilities for COMPASS web modules"""

from urllib.parse import quote, urlsplit, urlunsplit


def _sanitize_url(url):
    """Encode unsafe URL characters while preserving URL semantics"""
    parsed = urlsplit(url)
    path = quote(parsed.path, safe="/:@-._~!$&'()*+,;=")
    query = quote(parsed.query, safe="=&;%:@-._~!$&'()*+,;/?:")
    fragment = quote(parsed.fragment, safe="")
    return urlunsplit(
        (parsed.scheme, parsed.netloc, path, query, fragment)
    )
