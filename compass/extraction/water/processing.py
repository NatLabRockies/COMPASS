"""Water ordinance structured parsing class"""

import logging
from pathlib import Path

import pandas as pd
from elm import EnergyWizard
from elm.embed import ChunkAndEmbed
from elm.web.document import PDFDocument

from compass.utilities.enums import LLMTasks
from compass.exceptions import COMPASSRuntimeError


logger = logging.getLogger(__name__)


async def label_docs_no_legal_check(docs, **__):  # noqa: RUF029
    """Label documents with the "don't check for legal status" flag

    Parameters
    ----------
    docs : iterable of elm.web.document.PDFDocument
        Documents to label.

    Returns
    -------
    iterable of elm.web.document.PDFDocument
        Input docs with the "check_if_legal_doc" attribute set to False.
    """
    for doc in docs:
        doc.attrs["check_if_legal_doc"] = False
    return docs


async def build_corpus(docs, jurisdiction, model_configs, **__):
    """Build knowledge corpus for water rights extraction

    Parameters
    ----------
    docs : iterable of elm.web.document.PDFDocument
        Documents to build corpus from.
    jurisdiction : compass.utilities.location.Jurisdiction
        Jurisdiction being processed.
    model_configs : dict
        Dictionary of model configurations for various LLM tasks.

    Returns
    -------
    list or None
        List containing a single PDFDocument with the corpus, or None
        if no corpus could be built.

    Raises
    ------
    COMPASSRuntimeError
        If embeddings could not be generated.
    """
    model_config = model_configs.get(
        LLMTasks.EMBEDDING, model_configs[LLMTasks.DEFAULT]
    )
    _setup_endpoints(model_config)

    corpus = []
    for doc in docs:
        url = doc.attrs.get("source", "unknown source")
        logger.info("Embedding %r", url)
        obj = ChunkAndEmbed(
            doc.text,
            model=model_config.name,
            tokens_per_chunk=model_config.text_splitter_chunk_size,
            overlap=model_config.text_splitter_chunk_overlap,
            split_on="\n",
        )
        try:
            embeddings = await obj.run_async(rate_limit=3e4)
            if any(e is None for e in embeddings):
                msg = (
                    "Embeddings are ``None`` when building corpus for "
                    "water rights extraction!"
                )
                raise COMPASSRuntimeError(msg)  # noqa: TRY301

            corpus.append(
                pd.DataFrame(
                    {"text": obj.text_chunks.chunks, "embedding": embeddings}
                )
            )

        except Exception as e:  # noqa: BLE001
            logger.info("could not embed %r with error: %s", url, e)

    if len(corpus) == 0:
        logger.info(
            "No documents returned for %s, skipping", jurisdiction.full_name
        )
        return None

    corpus_doc = PDFDocument(
        ["water extraction context"], attrs={"corpus": pd.concat(corpus)}
    )
    return [corpus_doc]


async def extract_water_rights_ordinance_values(
    corpus_doc, parser_class, out_key, usage_tracker, model_config, **__
):
    """Extract ordinance values from a temporary vector store.

    Parameters
    ----------
    corpus_doc : elm.web.document.PDFDocument
        Document containing the vector store corpus.
    parser_class : type
        Class used to parse the vector store.
    out_key : str
        Key used to store extracted values in the document attributes.
    usage_tracker : compass.services.usage.UsageTracker
        Instance of the UsageTracker class used to track LLM usage.
    model_config : compass.llm.config.LLMConfig
        Model configuration used for LLM calls.

    Returns
    -------
    elm.web.document.PDFDocument
        Document with extracted ordinance values stored in attributes.
    """

    logger.debug("Building energy wizard")
    wizard = EnergyWizard(corpus_doc.attrs["corpus"], model=model_config.name)

    logger.debug("Calling parser class")
    parser = parser_class(
        wizard=wizard,
        location=corpus_doc.attrs["jurisdiction_name"],
        llm_service=model_config.llm_service,
        usage_tracker=usage_tracker,
        **model_config.llm_call_kwargs,
    )
    corpus_doc.attrs[out_key] = await parser.parse()
    return corpus_doc


def write_water_rights_data_to_disk(doc_infos, out_dir):
    """Write extracted water rights data to disk

    Parameters
    ----------
    doc_infos : list of dict
        List of dictionaries containing extracted document information
        and data file paths.
    out_dir : path-like
        Path to the output directory for the data.

    Returns
    -------
    int
        Number of unique water rights districts that information was
        found/written for.
    """
    db = []
    for doc_info in doc_infos:
        ord_db = pd.read_csv(doc_info["ord_db_fp"])
        if len(ord_db) == 0:
            continue
        ord_db["source"] = doc_info.get("source")

        year, *__ = doc_info.get("date") or (None, None, None)
        ord_db["ord_year"] = year if year is not None and year > 0 else None

        jurisdiction = doc_info["jurisdiction"]
        ord_db["WCD_ID"] = jurisdiction.code
        ord_db["county"] = jurisdiction.county
        ord_db["state"] = jurisdiction.state
        ord_db["subdivision"] = jurisdiction.subdivision_name
        ord_db["jurisdiction_type"] = jurisdiction.type

        db.append(ord_db)

    if not db:
        return 0

    db = pd.concat([df.dropna(axis=1, how="all") for df in db], axis=0)
    db.to_csv(Path(out_dir) / "water_rights.csv", index=False)
    return len(db["WCD_ID"].unique())


def _setup_endpoints(embedding_model_config):
    """Set proper URLS for elm classes"""
    ChunkAndEmbed.USE_CLIENT_EMBEDDINGS = True
    EnergyWizard.USE_CLIENT_EMBEDDINGS = True
    ChunkAndEmbed.EMBEDDING_MODEL = EnergyWizard.EMBEDDING_MODEL = (
        embedding_model_config.name
    )

    endpoint = embedding_model_config.client_kwargs["azure_endpoint"]
    ChunkAndEmbed.EMBEDDING_URL = endpoint
    ChunkAndEmbed.URL = endpoint
    EnergyWizard.EMBEDDING_URL = endpoint

    EnergyWizard.URL = "openai.azure.com"  # need to trigger Azure setup
