"""COMPASS Ordinance Threaded services"""

import json
import shutil
import asyncio
import hashlib
import logging
from pathlib import Path
from abc import abstractmethod
from tempfile import TemporaryDirectory
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

from elm.web.utilities import write_url_doc_to_file

from compass import COMPASS_DEBUG_LEVEL
from compass.services.base import Service
from compass.utilities import (
    LLM_COST_REGISTRY,
    extract_ord_year_from_doc_attrs,
    num_ordinances_in_doc,
)
from compass.pb import COMPASS_PB


logger = logging.getLogger(__name__)


def _cache_file_with_hash(doc, file_content, out_dir, make_name_unique=False):
    """Cache file and compute its hash"""
    cache_fp = write_url_doc_to_file(
        doc=doc,
        file_content=file_content,
        out_dir=out_dir,
        make_name_unique=make_name_unique,
    )
    return cache_fp, _compute_sha256(cache_fp)


def _compute_sha256(file_path):
    """Compute sha256 checksum for file on disk"""
    m = hashlib.sha256()
    m.update(Path(file_path).read_bytes())
    return f"sha256:{m.hexdigest()}"


def _move_file(doc, out_dir):
    """Move a file from a temp directory to an output directory"""
    cached_fp = doc.attrs.get("cache_fn")
    if cached_fp is None:
        return None

    cached_fp = Path(cached_fp)
    date = datetime.now().strftime("%Y_%m_%d")
    out_fn = doc.attrs.get("location_name", cached_fp.stem)
    out_fn = out_fn.replace(",", "").replace(" ", "_")
    out_fn = f"{out_fn}_downloaded_{date}"
    if not out_fn.endswith(cached_fp.suffix):
        out_fn = f"{out_fn}{cached_fp.suffix}"

    out_fp = Path(out_dir) / out_fn
    shutil.move(cached_fp, out_fp)
    return out_fp


def _write_cleaned_file(doc, out_dir):
    """Write cleaned ordinance text to directory"""
    location_name = doc.attrs.get("location_name")
    if location_name is None:
        return None

    out_dir = Path(out_dir)
    if COMPASS_DEBUG_LEVEL > 0:
        _write_interim_cleaned_files(doc, out_dir, location_name)

    key_to_fp = {
        "cleaned_ordinance_text": f"{location_name} Ordinance Summary.txt",
        "districts_text": f"{location_name} Districts.txt",
    }
    out_paths = []
    for key, fn in key_to_fp.items():
        cleaned_text = doc.attrs.get(key)
        if cleaned_text is None:
            continue

        out_fp = out_dir / fn
        out_fp.write_text(cleaned_text, encoding="utf-8")
        out_paths.append(out_fp)

    return out_paths


def _write_interim_cleaned_files(doc, out_dir, location_name):
    """Write intermediate output texts to file; helpful for debugging"""
    key_to_fp = {
        "ordinance_text": f"{location_name} Ordinance Original text.txt",
        "wind_energy_systems_text": f"{location_name} Wind Ordinance text.txt",
        "solar_energy_systems_text": (
            f"{location_name} Solar Ordinance text.txt"
        ),
        "permitted_use_text": (
            f"{location_name} Permitted Use Original text.txt"
        ),
        "permitted_use_only_text": (
            f"{location_name} Permitted Use Only text.txt"
        ),
    }
    for key, fn in key_to_fp.items():
        text = doc.attrs.get(key)
        if text is None:
            continue

        (out_dir / fn).write_text(text, encoding="utf-8")


def _write_ord_db(doc, out_dir):
    """Write parsed ordinance database to directory"""
    ord_db = doc.attrs.get("scraped_values")
    location_name = doc.attrs.get("location_name")

    if ord_db is None or location_name is None:
        return None

    out_fp = Path(out_dir) / f"{location_name} Ordinances.csv"
    ord_db.to_csv(out_fp, index=False)
    return out_fp


_PROCESSING_FUNCTIONS = {
    "move": _move_file,
    "write_clean": _write_cleaned_file,
    "write_db": _write_ord_db,
}


class ThreadedService(Service):
    """Service that contains a ThreadPoolExecutor instance"""

    def __init__(self, **kwargs):
        """

        Parameters
        ----------
        **kwargs
            Keyword-value argument pairs to pass to
            :class:`concurrent.futures.ThreadPoolExecutor`.
            By default, ``None``.
        """
        self._tpe_kwargs = kwargs or {}
        self.pool = None

    def acquire_resources(self):
        """Open thread pool and temp directory"""
        self.pool = ThreadPoolExecutor(**self._tpe_kwargs)

    def release_resources(self):
        """Shutdown thread pool and cleanup temp directory"""
        self.pool.shutdown(wait=True, cancel_futures=True)


class TempFileCache(ThreadedService):
    """Service that locally caches files downloaded from the internet"""

    def __init__(self, td_kwargs=None, tpe_kwargs=None):
        """

        Parameters
        ----------
        td_kwargs : dict, optional
            Keyword-value argument pairs to pass to
            :class:`tempfile.TemporaryDirectory`. By default, ``None``.
        tpe_kwargs : dict, optional
            Keyword-value argument pairs to pass to
            :class:`concurrent.futures.ThreadPoolExecutor`.
            By default, ``None``.
        """
        super().__init__(**(tpe_kwargs or {}))
        self._td_kwargs = td_kwargs or {}
        self._td = None

    @property
    def can_process(self):
        """bool: Always ``True`` (limiting is handled by asyncio)"""
        return True

    def acquire_resources(self):
        """Open thread pool and temp directory"""
        super().acquire_resources()
        self._td = TemporaryDirectory(**self._td_kwargs)

    def release_resources(self):
        """Shutdown thread pool and cleanup temp directory"""
        self._td.cleanup()
        super().release_resources()

    async def process(self, doc, file_content, make_name_unique=False):
        """Write URL doc to file asynchronously

        Parameters
        ----------
        doc : elm.web.document.Document
            Document containing meta information about the file. Must
            have a "source" key in the ``attrs`` dict containing the
            URL, which will be converted to a file name using
            :func:`compute_fn_from_url`.
        file_content : str or bytes
            File content, typically string text for HTML files and bytes
            for PDF file.
        make_name_unique : bool, optional
            Option to make file name unique by adding a UUID at the end
            of the file name. By default, ``False``.

        Returns
        -------
        Path
            Path to output file.
        """
        loop = asyncio.get_running_loop()
        cache_fp, checksum = await loop.run_in_executor(
            self.pool,
            _cache_file_with_hash,
            doc,
            file_content,
            self._td.name,
            make_name_unique,
        )
        logger.debug("Cached doc from %s", doc.attrs.get("source", "Unknown"))
        logger.trace("    ↪ checksum=%s", str(checksum))
        doc.attrs["checksum"] = checksum
        return cache_fp


class TempFileCachePB(TempFileCache):
    """Service that locally caches files downloaded from the internet"""

    async def process(self, doc, file_content, make_name_unique=False):
        """Write URL doc to file asynchronously

        Parameters
        ----------
        doc : elm.web.document.Document
            Document containing meta information about the file. Must
            have a "source" key in the ``attrs`` dict containing the
            URL, which will be converted to a file name using
            :func:`compute_fn_from_url`.
        file_content : str or bytes
            File content, typically string text for HTML files and bytes
            for PDF file.
        make_name_unique : bool, optional
            Option to make file name unique by adding a UUID at the end
            of the file name. By default, ``False``.

        Returns
        -------
        Path
            Path to output file.
        """
        out = await super().process(
            doc=doc,
            file_content=file_content,
            make_name_unique=make_name_unique,
        )
        location = asyncio.current_task().get_name()
        COMPASS_PB.update_download_task(location, advance=1)
        return out


class StoreFileOnDisk(ThreadedService):
    """Abstract service that manages the storage of a file on disk

    Storage can occur due to creation or a move of a file.
    """

    def __init__(self, out_dir, tpe_kwargs=None):
        """

        Parameters
        ----------
        out_dir : path-like
            Path to output directory where file should be stored.
        tpe_kwargs : dict, optional
            Keyword-value argument pairs to pass to
            :class:`concurrent.futures.ThreadPoolExecutor`.
            By default, ``None``.
        """
        super().__init__(**(tpe_kwargs or {}))
        self.out_dir = out_dir

    @property
    def can_process(self):
        """bool: Always ``True`` (limiting is handled by asyncio)"""
        return True

    async def process(self, doc):
        """Store file in out directory

        Parameters
        ----------
        doc : elm.web.document.Document
            Document containing meta information about the file. Must
            have relevant processing keys in the ``attrs`` dict,
            otherwise the file may not be stored in the output
            directory.

        Returns
        -------
        Path or None
            Path to output file, or `None` if no file was stored.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self.pool, _PROCESSING_FUNCTIONS[self._PROCESS], doc, self.out_dir
        )

    @property
    @abstractmethod
    def _PROCESS(self):  # noqa: N802
        """str: Key in `_PROCESSING_FUNCTIONS` defining the doc func"""
        raise NotImplementedError


class FileMover(StoreFileOnDisk):
    """Service that moves files to an output directory"""

    _PROCESS = "move"


class CleanedFileWriter(StoreFileOnDisk):
    """Service that writes cleaned text to a file"""

    _PROCESS = "write_clean"


class OrdDBFileWriter(StoreFileOnDisk):
    """Service that writes cleaned text to a file"""

    _PROCESS = "write_db"


class UsageUpdater(ThreadedService):
    """Service that updates usage info from a tracker into a file"""

    def __init__(self, usage_fp, tpe_kwargs=None):
        """

        Parameters
        ----------
        usage_fp : path-like
            Path to JSON file where usage should be tracked.
        tpe_kwargs : dict, optional
            Keyword-value argument pairs to pass to
            :class:`concurrent.futures.ThreadPoolExecutor`.
            By default, ``None``.
        """
        super().__init__(**(tpe_kwargs or {}))
        self.usage_fp = usage_fp
        self._is_processing = False

    @property
    def can_process(self):
        """bool: ``True`` if file not currently being written to"""
        return not self._is_processing

    async def process(self, tracker):
        """Add usage from tracker to file

        Any existing usage info in the file will remain unchanged
        EXCEPT for anything under the label of the input `tracker`,
        all of which will be replaced with info from the tracker itself.

        Parameters
        ----------
        tracker : elm.ods.services.usage.UsageTracker
            A usage tracker instance that contains usage info to be
            added to output file.
        """
        self._is_processing = True
        loop = asyncio.get_running_loop()
        out = await loop.run_in_executor(
            self.pool, _dump_usage, self.usage_fp, tracker
        )
        self._is_processing = False
        return out


class JurisdictionUpdater(ThreadedService):
    """Service that updates jurisdiction info into a file"""

    def __init__(self, jurisdiction_fp, tpe_kwargs=None):
        """

        Parameters
        ----------
        jurisdiction_fp : path-like
            Path to JSON file where jurisdictions should be tracked.
        tpe_kwargs : dict, optional
            Keyword-value argument pairs to pass to
            :class:`concurrent.futures.ThreadPoolExecutor`.
            By default, ``None``.
        """
        super().__init__(**(tpe_kwargs or {}))
        self.jurisdiction_fp = jurisdiction_fp
        self._is_processing = False

    @property
    def can_process(self):
        """bool: ``True`` if file not currently being written to"""
        return not self._is_processing

    async def process(self, county, doc, seconds_elapsed, usage_tracker=None):
        """Add usage from tracker to file

        Any existing usage info in the file will remain unchanged
        EXCEPT for anything under the label of the input `tracker`,
        all of which will be replaced with info from the tracker itself.

        Parameters
        ----------
        county : compass.utilities.location.Location
            County to record.
        doc : elm.web.document.Document
            Document containing meta information about the jurisdiction.
            Must have relevant processing keys in the ``attrs`` dict,
            otherwise the jurisdiction may not be recorded properly.
            If ``None``, the jurisdiction is assumed not to have been
            found.
        seconds_elapsed : int or float
            Total number of seconds it took to look for (and possibly
            parse) this document.
        usage_tracker : compass.services.usage.UsageTracker, optional
            Optional tracker instance to monitor token usage during
            LLM calls. By default, ``None``.
        """
        self._is_processing = True
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self.pool,
            _dump_jurisdiction_info,
            self.jurisdiction_fp,
            county,
            doc,
            seconds_elapsed,
            usage_tracker,
        )
        self._is_processing = False


def _dump_usage(fp, tracker):
    """Dump usage to an existing file"""
    if not Path(fp).exists():
        usage_info = {}
    else:
        with Path.open(fp, encoding="utf-8") as fh:
            usage_info = json.load(fh)

    if tracker is not None:
        tracker.add_to(usage_info)

        with Path.open(fp, "w", encoding="utf-8") as fh:
            json.dump(usage_info, fh, indent=4)

    return usage_info


def _dump_jurisdiction_info(fp, county, doc, seconds_elapsed, usage_tracker):
    """Dump jurisdiction info to an existing file"""
    if not Path(fp).exists():
        jurisdiction_info = {"jurisdictions": []}
    else:
        with Path.open(fp, encoding="utf-8") as fh:
            jurisdiction_info = json.load(fh)

    new_info = {
        "full_name": county.full_name,
        "county": county.name,
        "state": county.state,
        "subdivision": None,
        "jurisdiction_type": None,
        "FIPS": county.fips,
        "found": False,
        "total_time": seconds_elapsed,
        "total_time_string": str(timedelta(seconds=seconds_elapsed)),
        "cost": None,
        "documents": None,
    }
    if usage_tracker is not None:
        cost = _compute_jurisdiction_cost(usage_tracker)
        new_info["cost"] = cost or None

    if num_ordinances_in_doc(doc) > 0:
        new_info["found"] = True
        new_info["documents"] = [_compile_doc_info(doc)]

    jurisdiction_info["jurisdictions"].append(new_info)
    with Path.open(fp, "w", encoding="utf-8") as fh:
        json.dump(jurisdiction_info, fh, indent=4)


def _compile_doc_info(doc):
    """Put together meta information about a single document"""
    return {
        "source": doc.attrs.get("source"),
        "ord_year": extract_ord_year_from_doc_attrs(doc.attrs),
        "ord_filename": Path(doc.attrs.get("out_fp", "Unknown")).name,
        "num_pages": len(doc.pages),
        "checksum": doc.attrs.get("checksum"),
    }


def _compute_jurisdiction_cost(usage_tracker):
    """Compute total cost from total tracked usage"""

    total_cost = 0
    for model, total_usage in usage_tracker.totals.items():
        model_costs = LLM_COST_REGISTRY.get(model, {})
        total_cost += (
            total_usage.get("prompt_tokens", 0)
            / 1e6
            * model_costs.get("prompt", 0)
        )
        total_cost += (
            total_usage.get("response_tokens", 0)
            / 1e6
            * model_costs.get("response", 0)
        )

    return total_cost
