"""Base COMPASS utility functions"""

from pathlib import Path
from copy import deepcopy
from functools import cached_property

from elm.web.search.run import SEARCH_ENGINE_OPTIONS


def title_preserving_caps(string):
    """Convert string to title case, preserving existing capitalization

    Parameters
    ----------
    string : str
        Input string potentially containing capitalized words.

    Returns
    -------
    str
        String converted to title case, preserving existing
        capitalization.
    """
    return " ".join(map(_cap, string.split(" ")))


class WebSearchParams:
    """Helper class to store web search params"""

    def __init__(
        self,
        num_urls_to_check_per_jurisdiction=5,
        max_num_concurrent_browsers=10,
        max_num_concurrent_website_searches=None,
        url_ignore_substrings=None,
        pytesseract_exe_fp=None,
        search_engines=None,
    ):
        """

        Parameters
        ----------
        num_urls_to_check_per_jurisdiction : int, optional
            Number of unique Google search result URLs to check for each
            jurisdiction when attempting to locate ordinance documents.
            By default, ``5``.
        max_num_concurrent_browsers : int, optional
            Maximum number of browser instances to launch concurrently
            for retrieving information from the web. Increasing this
            value too much may lead to timeouts or performance issues on
            machines with limited resources. By default, ``10``.
        max_num_concurrent_website_searches : int, optional
            Maximum number of website searches allowed to run
            simultaneously. Increasing this value can speed up searches,
            but may lead to timeouts or performance issues on machines
            with limited resources. By default, ``10``.
        url_ignore_substrings : list of str, optional
            A list of substrings that, if found in any URL, will cause
            the URL to be excluded from consideration. This can be used
            to specify particular websites or entire domains to ignore.
            For example::

                url_ignore_substrings = [
                    "wikipedia",
                    "nrel.gov",
                    "www.co.delaware.in.us/documents/1649699794_0382.pdf",
                ]

            The above configuration would ignore all `wikipedia`
            articles, all websites on the NREL domain, and the specific
            file located at
            `www.co.delaware.in.us/documents/1649699794_0382.pdf`.
            By default, ``None``.
        pytesseract_exe_fp : path-like, optional
            Path to the `pytesseract` executable. If specified, OCR will
            be used to extract text from scanned PDFs using Google's
            Tesseract. By default ``None``.
        search_engines : list, optional
            A list of dictionaries, where each dictionary contains
            information about a search engine class that should be used
            for the document retrieval process. Each dictionary should
            contain at least the key ``"se_name"``, which should
            correspond to one of the search engine class names from
            :obj:`elm.web.search.run.SEARCH_ENGINE_OPTIONS`. The rest of
            the keys in the dictionary should contain keyword-value
            pairs to be used as parameters to initialize the search
            engine class (things like API keys and configuration
            options; see the ELM documentation for details on search
            engine class parameters). The list should be ordered by
            search engine preference - the first search engine
            parameters will be used to submit the queries initially,
            then any subsequent search engine listings  will be used as
            fallback (in order that they appear). If ``None``, then all
            default configurations for the search engines (along with
            the fallback order) are used. By default, ``None``.
        """
        self.num_urls_to_check_per_jurisdiction = (
            num_urls_to_check_per_jurisdiction
        )
        self.max_num_concurrent_browsers = max_num_concurrent_browsers
        self.max_num_concurrent_website_searches = (
            max_num_concurrent_website_searches
        )
        self.url_ignore_substrings = url_ignore_substrings
        self.pytesseract_exe_fp = pytesseract_exe_fp
        self._search_engines_input = search_engines

    @cached_property
    def se_kwargs(self):
        """dict: Extra search engine kwargs to pass to ELM"""
        if not self._search_engines_input:
            return {}

        search_engines = []
        extra_kwargs = {}
        for se_params in self._search_engines_input:
            params = deepcopy(se_params)
            se_name = params.pop("se_name")
            search_engines.append(se_name)
            extra_kwargs[SEARCH_ENGINE_OPTIONS[se_name].kwg_key_name] = params

        extra_kwargs["search_engines"] = search_engines
        return extra_kwargs


class Directories:
    """Helper class to store directories used in COMPASS run"""

    def __init__(
        self,
        out,
        logs=None,
        clean_files=None,
        ordinance_files=None,
        jurisdiction_dbs=None,
    ):
        """

        Parameters
        ----------
        out : path-like
            Output directory for COMPASS run.
        logs : path-like, optional
            Directory for storing logs. If not specified, defaults to
            ``out/logs``. By default, ``None``.
        clean_files : path-like, optional
            Directory for storing cleaned ordinance files. If not
            specified, defaults to ``out/cleaned_text``.
            By default, ``None``.
        ordinance_files : path-like, optional
            Directory for storing ordinance files. If not specified,
            defaults to ``out/ordinance_files``.
            By default, ``None``.
        jurisdiction_dbs : path-like, optional
            Directory for storing jurisdiction databases. If not
            specified, defaults to ``out/jurisdiction_dbs``.
            By default, ``None``
        """
        self.out = _full_path(out)
        self.logs = _full_path(logs) if logs else self.out / "logs"
        self.clean_files = (
            _full_path(clean_files)
            if clean_files
            else self.out / "cleaned_text"
        )
        self.ordinance_files = (
            _full_path(ordinance_files)
            if ordinance_files
            else self.out / "ordinance_files"
        )
        self.jurisdiction_dbs = (
            _full_path(jurisdiction_dbs)
            if jurisdiction_dbs
            else self.out / "jurisdiction_dbs"
        )

    def __iter__(self):
        """Iterate over all directories"""
        yield self.out
        yield self.logs
        yield self.clean_files
        yield self.ordinance_files
        yield self.jurisdiction_dbs

    def make_dirs(self):
        """Create all directories if they do not exist"""
        for folder in self:
            folder.mkdir(exist_ok=True, parents=True)


def _cap(word):
    """Capitalize first letter of the word"""
    return "".join([word[0].upper(), word[1:]])


def _full_path(in_path):
    """Expand and resolve input path"""
    return Path(in_path).expanduser().resolve()
