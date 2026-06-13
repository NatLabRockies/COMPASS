"""Shared URL utilities for COMPASS web modules"""

from urllib.parse import quote, urlsplit, urlunsplit


_PATH_SAFE_CHARS = "/:@-._~!$&'()*+,;=%"
_QUERY_SAFE_CHARS = "=&;%:@-._~!$&'()*+,;/?"


def sanitize_url(url):
    """Encode unsafe URL characters while preserving URL semantics

    Parameters
    ----------
    url : str
        URL string that may include unsafe characters such as spaces.

    Returns
    -------
    str
        URL with path, query, and fragment percent-encoded.
    """
    parsed = urlsplit(url)
    path = quote(parsed.path, safe=_PATH_SAFE_CHARS)
    query = quote(parsed.query, safe=_QUERY_SAFE_CHARS)
    fragment = quote(parsed.fragment, safe="")
    return urlunsplit((parsed.scheme, parsed.netloc, path, query, fragment))


def base_website_url(url):
    """Return the scheme and netloc portion of a website URL

    Parameters
    ----------
    url : str
        URL string that may include a path, query string, or fragment.

    Returns
    -------
    str
        Canonical website root URL consisting of the original scheme
        and netloc with a trailing slash. If the URL is missing a
        scheme or netloc, the input is returned unchanged.
    """
    parsed = urlsplit(url)
    if not parsed.scheme or not parsed.netloc:
        return url

    return urlunsplit((parsed.scheme, parsed.netloc, "/", "", ""))
