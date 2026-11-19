"""Shared URL utilities for COMPASS web modules"""

import re
import logging
from urllib.parse import parse_qs, quote, urlsplit, urlunsplit, unquote

from elm.web.utilities import clean_json_escaped_url


logger = logging.getLogger(__name__)
_PATH_SAFE_CHARS = "/:@-._~!$&'()*+,;=%"
_QUERY_SAFE_CHARS = "=&;%:@-._~!$&'()*+,;/?"
_TRACKING_PARAMETERS = {"gclid", "fbclid"}
_DOC_VIEWER_QUERY_KEYS = ("document", "file", "url", "src", "pdf", "href")
"""Query-string keys that commonly hold a viewer's real document URL"""

_GOOGLE_DRIVE_HOSTS = {"drive.google.com", "docs.google.com"}
"""Hosts serving Google Drive file viewers"""

_GOOGLE_DRIVE_FILE_PATH = re.compile(r"/file/d/([\w-]+)")
"""Path of a Drive file viewer, capturing the file ID"""

_DOC_SUFFIXES = (".pdf", ".doc", ".docx", ".rtf")
"""Document extensions worth unwrapping a viewer link for"""


class URLPartFilter:
    """Match URL parts with whitelist precedence"""

    def __init__(self, blacklist=None, whitelist=None):
        """

        Parameters
        ----------
        blacklist : iterable of str, optional
            URL parts that exclude matching URLs. By default, ``None``.
        whitelist : iterable of str, optional
            URL parts that override blacklist matches. By default,
            ``None``.
        """
        self.blacklist = _normalize_url_parts(blacklist)
        self.whitelist = _normalize_url_parts(whitelist)

    def blacklist_match(self, url):
        """Return the first blacklist match unless URL is whitelisted

        Parameters
        ----------
        url : str
            URL string to check against the blacklist.

        Returns
        -------
        str or None
            The first matching blacklist part if found and not
            whitelisted, otherwise ``None``.
        """
        url = url.casefold()
        if self.is_whitelisted(url):
            return None
        return next((part for part in self.blacklist if part in url), None)

    def is_whitelisted(self, url):
        """Check whether any whitelist part occurs in a URL

        Parameters
        ----------
        url : str
            URL string to check against the whitelist.

        Returns
        -------
        bool
            ``True`` if any whitelist part occurs in the URL, otherwise
            ``False``.

        """
        url = url.casefold()
        return any(part in url for part in self.whitelist)


def sanitize_url(url):
    """Encode unsafe URL characters and drop what does not identify it

    Two links to the same page give the same result, so it can be used
    both to fetch a page and to tell whether it was already seen.
    Sanitizing a URL a second time changes nothing.

    Parameters
    ----------
    url : str
        URL string that may include unsafe characters such as spaces.

    Returns
    -------
    str
        URL with its path, query, and fragment percent-encoded, its
        scheme and host in lower case, and tracking parameters
        (``utm_*``, ``gclid``, ``fbclid``) removed. The fragment of a
        PDF link, such as ``#page=3``, is dropped. Other fragments are
        kept, since code libraries use them to address a page
        (``#/county/zoning``).
    """
    parsed = urlsplit(clean_json_escaped_url(url.strip()))
    path = quote(parsed.path, safe=_PATH_SAFE_CHARS)
    query = "&".join(
        part
        for part in quote(parsed.query, safe=_QUERY_SAFE_CHARS).split("&")
        if not _is_tracking_parameter(part)
    )
    # Decoded before it is encoded, so an escaped fragment is not
    # escaped a second time, and its "/" separators stay readable
    fragment = quote(unquote(parsed.fragment), safe=_QUERY_SAFE_CHARS)
    if path.lower().endswith(".pdf"):
        fragment = ""
    if parsed.netloc and not path:
        path = "/"
    return urlunsplit(
        (parsed.scheme.lower(), parsed.netloc.lower(), path, query, fragment)
    )


def _is_tracking_parameter(part):
    """Whether a ``name=value`` query part only tracks a visit"""
    name = unquote(part.split("=", 1)[0]).lower()
    return name.startswith("utm_") or name in _TRACKING_PARAMETERS


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


def normalize_domain(url):
    """Return a comparable domain string for a URL or empty string

    Parameters
    ----------
    url : str
        URL string to extract the domain from.

    Returns
    -------
    str
        Normalized domain string, lowercased and without www prefix.
    """
    parsed = urlsplit(url.strip())
    domain = parsed.netloc or parsed.path.partition("/")[0]
    domain = domain.partition("@")[2] or domain
    domain = domain.partition(":")[0].casefold().strip()
    if domain.startswith("www."):
        return domain[4:]
    return domain


def _normalize_url_parts(parts):
    """Normalize non-empty URL parts for matching"""
    return [part.casefold() for part in parts or [] if part]


def _google_drive_download_url(href):
    """Download URL of a file behind a Google Drive viewer link, if any

    ``drive.google.com/file/d/<ID>/view`` returns Drive's viewer page,
    whose document pages are images loaded by script, so its text is
    only the viewer's menus. ``uc?export=download&id=<ID>`` returns the
    file itself.
    """
    parts = urlsplit(href)
    if parts.netloc.casefold() not in _GOOGLE_DRIVE_HOSTS:
        return None
    match = _GOOGLE_DRIVE_FILE_PATH.match(parts.path)
    if match is None:
        return None
    return f"https://drive.google.com/uc?export=download&id={match[1]}"


def unwrap_document_viewer_url(href):
    """Resolve a document-viewer link to the document it wraps

    Some sites link to documents through a viewer page that carries the
    real file URL in its query string, e.g.::

        /archival-document?document=https://.../ordinance.pdf&title=...

    Fetching the viewer returns the page's navigation chrome rather than
    the document (about 1 KB of menus instead of the ordinance), so the
    embedded URL is used when one is present.

    Parameters
    ----------
    href : str
        URL that may wrap another document URL.

    Returns
    -------
    str
        The embedded document URL, or `href` unchanged if there is none.
    """
    if drive_file := _google_drive_download_url(href):
        logger.debug("Unwrapped Google Drive link %s -> %s", href, drive_file)
        return drive_file

    params = parse_qs(urlsplit(href).query)
    keys = [key for key in _DOC_VIEWER_QUERY_KEYS if key in params]
    keys += [key for key in params if key not in _DOC_VIEWER_QUERY_KEYS]
    values = [value for key in keys for value in params[key]]
    embedded = next(filter(is_document_url, values), None)
    if embedded is None:
        return href
    logger.debug("Unwrapped viewer link %s -> %s", href, embedded)
    return embedded


def is_document_url(value):
    """Whether a query value is a web link to a document file"""
    parts = urlsplit(value)
    return parts.scheme in {
        "http",
        "https",
    } and parts.path.casefold().endswith(_DOC_SUFFIXES)
