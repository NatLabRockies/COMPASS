"""Persistence for collected documents"""

import json
import asyncio
from pathlib import Path
from warnings import warn
from datetime import datetime, UTC

from compass.services.threaded import (
    FileMover,
    ParsedFileWriter,
    TempFileCacheCopier,
    GenericFuncRunner,
)
from compass.services.cpu import read_docling_local_file
from compass.utilities.io import load_config
from compass.utilities.io import resolve_all_paths
from compass.utilities.parsing import convert_paths_to_strings, is_pdf_doc
from compass.warn import COMPASSWarning
from compass.exceptions import COMPASSFileNotFoundError, COMPASSValueError


COLLECTION_MANIFEST_FILENAME = "collection_manifest.json"


def build_collection_manifest(
    tech, jurisdictions, time_start_utc, num_jurisdictions_searched
):
    """Build the serialized collection manifest payload

    Parameters
    ----------
    tech : str
        Technology specified in the pipeline request, included in the
        manifest for compatibility validation when loading.
    jurisdictions : dict
        Dictionary mapping jurisdiction full names to serialized
        collection metadata for each jurisdiction, including
        jurisdiction identifiers and the persisted document records.
    time_start_utc : datetime.datetime
        UTC datetime when the collection process started, used to
        calculate elapsed time for the manifest metadata.
    num_jurisdictions_searched : int
        Number of jurisdictions that were searched during the collection
        process, included in the manifest for informational purposes.

    Returns
    -------
    dict
        Collection manifest as a dictionary, ready to be serialized and
        written to disk.
    """
    time_end_utc = datetime.now(UTC)
    time_elapsed = time_end_utc - time_start_utc
    return {
        "tech": tech,
        "time_start_utc": time_start_utc.isoformat(),
        "time_end_utc": time_end_utc.isoformat(),
        "total_time": time_elapsed.total_seconds(),
        "total_time_string": str(time_elapsed),
        "num_jurisdictions_searched": num_jurisdictions_searched,
        "num_jurisdictions_found": len(jurisdictions),
        "jurisdictions": jurisdictions,
    }


async def write_collection_manifest(manifest_dir, collection_manifest):
    """Write a collection manifest to disk

    Parameters
    ----------
    manifest_dir : path-like
        Path to the directory where the manifest should be written.
    collection_manifest : dict
        Dictionary containing collection manifest information to be
        serialized and written to disk.

    Returns
    -------
    pathlib.Path
        Path to the written manifest file.
    """
    return await GenericFuncRunner.call(
        _write_collection_manifest, manifest_dir, collection_manifest
    )


async def write_collection_manifest_shard(shard_dir, collection_info):
    """Write one jurisdiction collection manifest shard to disk

    Parameters
    ----------
    shard_dir : path-like
        Directory where the jurisdiction shard JSON should be written.
    collection_info : dict
        Serialized collection metadata for one jurisdiction.

    Returns
    -------
    pathlib.Path
        Path to the written shard file.
    """
    return await GenericFuncRunner.call(
        _write_collection_manifest_shard, shard_dir, collection_info
    )


async def load_collection_manifest(manifest_fp, expected_tech):
    """Load a collection manifest from disk

    Parameters
    ----------
    manifest_fp : path-like
        Path to the collection manifest file to be loaded.
    expected_tech : str
        Technology specified in the pipeline request, used to validate
        compatibility with the manifest.

    Returns
    -------
    dict
        Loaded collection manifest as a dictionary.
    """
    return await GenericFuncRunner.call(
        _load_collection_manifest, manifest_fp, expected_tech
    )


def _write_collection_manifest(manifest_dir, collection_manifest):
    """Write a collection manifest to disk"""
    manifest_fp = Path(manifest_dir) / COLLECTION_MANIFEST_FILENAME
    manifest_fp.write_text(
        json.dumps(convert_paths_to_strings(collection_manifest), indent=4),
        encoding="utf-8",
    )
    return manifest_fp


def _write_collection_manifest_shard(shard_dir, collection_info):
    """Write one jurisdiction collection manifest shard to disk"""
    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_fp = shard_dir / _collection_manifest_shard_filename(collection_info)
    shard_fp.write_text(
        json.dumps(convert_paths_to_strings(collection_info), indent=4),
        encoding="utf-8",
    )
    return shard_fp


def _load_collection_manifest(manifest_fp, expected_tech):
    """Load a collection manifest from disk"""
    try:
        manifest = load_config(manifest_fp, file_name="Collection manifest")
    except COMPASSFileNotFoundError:
        manifest = _load_collection_manifest_from_shards(
            manifest_fp, expected_tech
        )
        if manifest is None:
            raise

        msg = (
            f"Collection manifest file '{manifest_fp}' is missing; rebuilding "
            "collection manifest from jurisdiction shard files"
        )
        warn(msg, COMPASSWarning)

    _validate_collection_manifest(manifest, expected_tech)
    return manifest


async def persist_documents(jurisdiction, collected_docs, *, relative_to=None):
    """Persist deduplicated documents for one jurisdiction

    Parameters
    ----------
    jurisdiction : compass.utilities.jurisdictions.Jurisdiction
        Jurisdiction whose deduplicated documents will be persisted and
        serialized into collection metadata.
    collected_docs : \
            compass.pipeline.collection.dedupe.DocumentDeDuplicator
        Deduplicated document collection containing ``{"doc",
        "from_steps"}`` entries for each persisted document.
    relative_to : path-like, optional
        Base path used to store ``source_fp`` and ``parsed_fp`` as
        relative paths when possible. By default, ``None``.

    Returns
    -------
    dict
        Serialized collection metadata for the jurisdiction, including
        jurisdiction identifiers and the persisted document records.
    """
    tasks = []
    for index, info in enumerate(collected_docs.values, start=1):
        task = asyncio.create_task(
            _persist_doc(
                info["doc"],
                out_fn=f"{jurisdiction.full_name}_{index}",
                from_steps=info["from_steps"],
                relative_to=relative_to,
            ),
            name=jurisdiction.full_name,
        )
        tasks.append(task)

    documents = await asyncio.gather(*tasks)
    return {
        "full_name": jurisdiction.full_name,
        "county": jurisdiction.county,
        "state": jurisdiction.state,
        "subdivision": jurisdiction.subdivision_name,
        "jurisdiction_type": jurisdiction.type,
        "FIPS": jurisdiction.code,
        "documents": documents,
    }


async def load_collected_docs(collection_info, *, task_name):
    """Load all docs for one jurisdiction from collection info

    Parameters
    ----------
    collection_info : dict
        Persisted collection metadata for one jurisdiction, including
        a ``documents`` list of serialized document records.
    task_name : str
        Task name applied to each asynchronous document-loading task.

    Returns
    -------
    list
        Loaded document objects in the same order as the persisted
        ``documents`` entries.
    """
    tasks = [
        asyncio.create_task(_load_single_doc(doc_info), name=task_name)
        for doc_info in collection_info.get("documents") or []
    ]

    return await asyncio.gather(*tasks)


def _validate_collection_manifest(manifest, tech):
    """Validate manifest version and tech compatibility"""
    manifest_tech = manifest.get("tech")
    if manifest_tech and manifest_tech != tech:
        msg = (
            f"Collection manifest tech ({manifest_tech}) does not "
            f"match specified tech ({tech})"
        )
        raise COMPASSValueError(msg)


async def _load_single_doc(doc_info):
    """Load one document from persisted collection artifacts"""
    fp = doc_info.get("parsed_fp")
    if fp is None:
        msg = (
            "Parsed file path ('parsed_fp') is required to load a "
            "collected document, but it is missing from the following "
            f"doc info:\n{doc_info}\nSkipping..."
        )
        warn(msg, COMPASSWarning)
        return None

    doc, *__ = await read_docling_local_file(fp)
    doc.attrs.update(doc_info)
    doc.remove_comments = False
    doc.attrs["cache_fn"] = await TempFileCacheCopier.call(doc)
    return doc


async def _persist_doc(doc, out_fn, from_steps, relative_to):
    """Persist one collected document and its parsed text"""
    await _move_file_to_collection_dir(doc, out_fn, relative_to)
    await _persist_parsed_text(doc, out_fn, relative_to)
    return _serialize_collection_doc_info(doc, from_steps)


async def _move_file_to_collection_dir(doc, out_fn, relative_to):
    """Move a source file to the collection output directory"""
    out_fp = await FileMover.call(doc, out_fn, "downloaded")
    if relative_to is not None and out_fp is not None:
        out_fp = _make_relative(out_fp, relative_to)
    doc.attrs["source_fp"] = out_fp


async def _persist_parsed_text(doc, out_fn, relative_to):
    """Write parsed text for a collected document"""
    out_fp = await ParsedFileWriter.call(doc, out_fn)
    if relative_to is not None and out_fp is not None:
        out_fp = _make_relative(out_fp, relative_to)
    doc.attrs["parsed_fp"] = out_fp


def _make_relative(fp, relative_to):
    """Make a file path relative to another path when possible"""
    try:
        return fp.relative_to(relative_to)
    except ValueError:
        msg = (
            f"Could not make path {fp} relative to {relative_to}; using "
            "absolute path instead"
        )
        warn(msg, COMPASSWarning)
        return fp


def _serialize_collection_doc_info(doc, from_steps):
    """Serialize a collected document for manifest storage"""
    serialized = dict(doc.attrs)
    serialized.pop("cache_fn", None)
    serialized.pop("cleaned_fps", None)
    serialized.setdefault("check_correct_jurisdiction", True)
    serialized.update(
        {
            "is_pdf": doc.attrs.get("is_pdf", is_pdf_doc(doc)),
            "num_pages": doc.attrs.get("num_pages", len(doc.pages)),
            "from_steps": from_steps,
        }
    )
    return serialized


def _collection_manifest_shard_filename(collection_info):
    """Build a deterministic shard filename for one jurisdiction"""
    identifier = _clean_shard_name_part(collection_info.get("FIPS"))
    full_name = _clean_shard_name_part(collection_info.get("full_name"))
    name_parts = [part for part in (identifier, full_name) if part]
    base_name = "_".join(name_parts) or "jurisdiction"
    return f"{base_name}_collection_manifest.json"


def _clean_shard_name_part(value):
    """Normalize one shard filename component"""
    if value is None:
        return ""

    value = str(value).strip()
    for old, new in (("/", "-"), ("\\", "-"), (":", "-")):
        value = value.replace(old, new)
    return value


def _load_collection_manifest_from_shards(manifest_fp, expected_tech):
    """Rebuild a collection manifest from jurisdiction shard files"""
    manifest_dir = Path(manifest_fp).expanduser().resolve().parent
    shard_fps = sorted(manifest_dir.rglob("*_collection_manifest.json"))
    if not shard_fps:
        return None

    jurisdictions = []
    for shard_fp in shard_fps:
        collection_info = load_config(
            shard_fp,
            resolve_paths=False,
            file_name="Collection manifest shard",
        )
        jurisdictions.append(resolve_all_paths(collection_info, manifest_dir))

    return build_collection_manifest(expected_tech, jurisdictions)
