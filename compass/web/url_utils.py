"""Shared URL utilities for COMPASS web modules"""

from urllib.parse import urlparse, urlunparse


def _sanitize_url(url):
    """Encode spaces in a URL path; leave query string intact"""
    parsed = urlparse(url)
    path = parsed.path
    safe_path = path.replace(" ", "%20") if " " in path else path
    return urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            safe_path,
            parsed.params,
            parsed.query,
            parsed.fragment,
        )
    )
