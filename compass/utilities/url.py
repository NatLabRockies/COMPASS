"""Shared URL utilities for COMPASS web modules"""

from urllib.parse import quote, urlsplit, urlunsplit

from elm.web.utilities import clean_json_escaped_url


_PATH_SAFE_CHARS = "/:@-._~!$&'()*+,;=%"
_QUERY_SAFE_CHARS = "=&;%:@-._~!$&'()*+,;/?"


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
        """Return the first blacklist match not covered by the whitelist

        Parameters
        ----------
        url : str
            URL string to check against the blacklist.

        Returns
        -------
        str or None
            The first matching blacklist part not contained in a
            matching whitelist part, otherwise ``None``.
        """
        url = url.casefold()
        whitelist_matches = [part for part in self.whitelist if part in url]
        return next(
            (
                part
                for part in self.blacklist
                if part in url
                and not any(part in match for match in whitelist_matches)
            ),
            None,
        )

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
    url = clean_json_escaped_url(url)
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
