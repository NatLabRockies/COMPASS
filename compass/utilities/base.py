"""Base COMPASS utility functions"""

from pathlib import Path


def title_preserving_caps(string):
    """Convert text to title case while keeping intentional capitals

    Parameters
    ----------
    string : str
        Input text that may already contain capitalized acronyms or
        proper nouns.

    Returns
    -------
    str
        Title-cased string in which words containing existing uppercase
        characters retain their capitalization.

    Examples
    --------
    >>> title_preserving_caps("NLR solar ordinance")
    'NLR Solar Ordinance'
    """
    return " ".join(map(_cap, string.split(" ")))


class Directories:
    """Encapsulate filesystem locations used by a COMPASS run

    The helper centralizes directory computations so downstream code
    can rely on fully resolved :class:`pathlib.Path` instances for
    logging, cleaned text, downloaded ordinances, and intermediate
    databases.

    Notes
    -----
    All provided paths are expanded to absolute form when the class is
    instantiated, guaranteeing consistent behavior across relative and
    user-expanded paths.
    """

    def __init__(
        self,
        out,
        logs=None,
        clean_files=None,
        ordinance_files=None,
        jurisdiction_dbs=None,
        collect_only=False,
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
            else self.out / ("parsed_docs" if collect_only else "cleaned_text")
        )
        self.ordinance_files = (
            _full_path(ordinance_files)
            if ordinance_files
            else self.out
            / ("source_docs" if collect_only else "ordinance_files")
        )
        self.jurisdiction_dbs = (
            _full_path(jurisdiction_dbs)
            if jurisdiction_dbs
            else self.out
            / ("manifest_shards" if collect_only else "jurisdiction_dbs")
        )

    def __iter__(self):
        """Yield managed directory paths in canonical order

        Yields
        ------
        pathlib.Path
            Each of the managed directories in the following order:
            out, logs, clean_files, ordinance_files, jurisdiction_dbs.
        """
        yield self.out
        yield self.logs
        yield self.clean_files
        yield self.ordinance_files
        yield self.jurisdiction_dbs

    def make_dirs(self):
        """Create the managed directories if they do not exist"""
        for folder in self:
            folder.mkdir(exist_ok=True, parents=True)


def _cap(word):
    """Capitalize the first character of ``word``; preserve the rest"""
    return "".join([word[0].upper(), word[1:]])


def _full_path(in_path):
    """Resolve an input path to an absolute :class:`pathlib.Path`"""
    return Path(in_path).expanduser().resolve()
