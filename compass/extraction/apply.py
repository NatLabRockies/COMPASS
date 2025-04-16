"""Ordinance function to apply ordinance extraction on a document"""

import logging
from warnings import warn

from compass.llm import StructuredLLMCaller
from compass.extraction.date import DateExtractor
from compass.validation import (
    ParseChunksWithMemory,
    LegalTextValidator,
    parse_by_chunks,
)
from compass.warn import COMPASSWarning


logger = logging.getLogger(__name__)


async def check_for_ordinance_info(
    doc,
    model_config,
    heuristic,
    ordinance_text_collector_class,
    permitted_use_text_collector_class=None,
    usage_tracker=None,
):
    """Parse a single document for ordinance information

    Parameters
    ----------
    doc : elm.web.document.BaseDocument
        A document potentially containing ordinance information. Note
        that if the document's attrs contains the
        ``"contains_ord_info"`` key, it will not be processed. To force
        a document to be processed by this function, remove that key
        from the documents attrs.
    text_splitter : obj
        Instance of an object that implements a `split_text` method.
        The method should take text as input (str) and return a list
        of text chunks. Langchain's text splitters should work for this
        input.
    usage_tracker : compass.services.usage.UsageTracker, optional
        Optional tracker instance to monitor token usage during
        LLM calls. By default, ``None``.

    Returns
    -------
    elm.web.document.BaseDocument
        Document that has been parsed for ordinance text. The results of
        the parsing are stored in the documents attrs. In particular,
        the attrs will contain a ``"contains_ord_info"`` key that
        will be set to ``True`` if ordinance info was found in the text,
        and ``False`` otherwise. If ``True``, the attrs will also
        contain a ``"date"`` key containing the most recent date that
        the ordinance was enacted (or a tuple of `None` if not found),
        and an ``"ordinance_text"`` key containing the ordinance text
        snippet. Note that the snippet may contain other info as well,
        but should encapsulate all of the ordinance text.
    """
    if "contains_ord_info" in doc.attrs:
        return doc

    llm_caller = StructuredLLMCaller(
        llm_service=model_config.llm_service,
        usage_tracker=usage_tracker,
        **model_config.llm_call_kwargs,
    )
    chunks = model_config.text_splitter.split_text(doc.text)
    chunk_parser = ParseChunksWithMemory(llm_caller, chunks, num_to_recall=2)
    legal_text_validator = LegalTextValidator()

    ordinance_text_collector = ordinance_text_collector_class()
    callbacks = [ordinance_text_collector.check_chunk]
    if permitted_use_text_collector_class is not None:
        permitted_use_text_collector = permitted_use_text_collector_class()
        callbacks.append(permitted_use_text_collector.check_chunk)

    doc.attrs["is_legal_text"] = await parse_by_chunks(
        chunk_parser,
        heuristic,
        legal_text_validator,
        callbacks=callbacks,
        min_chunks_to_process=3,
    )

    doc.attrs["contains_ord_info"] = ordinance_text_collector.contains_ord_info
    if doc.attrs["contains_ord_info"]:
        doc.attrs["ordinance_text"] = ordinance_text_collector.ordinance_text
        logger.debug_to_file(
            "Ordinance text for %s is:\n%s",
            doc.attrs.get("source", "unknown source"),
            doc.attrs["ordinance_text"],
        )

    if permitted_use_text_collector_class is not None:
        doc.attrs["contains_district_info"] = (
            permitted_use_text_collector.contains_district_info
        )
        if doc.attrs["contains_district_info"]:
            doc.attrs["permitted_use_text"] = (
                permitted_use_text_collector.permitted_use_district_text
            )
            logger.debug_to_file(
                "Permitted use text for %s is:\n%s",
                doc.attrs.get("source", "unknown source"),
                doc.attrs["permitted_use_text"],
            )

    return doc


async def extract_date(doc, model_config, usage_tracker=None):
    """Parse a single document for date information

    Parameters
    ----------
    doc : elm.web.document.BaseDocument
        A document potentially containing date information.
    usage_tracker : compass.services.usage.UsageTracker, optional
        Optional tracker instance to monitor token usage during
        LLM calls. By default, ``None``.

    Returns
    -------
    elm.web.document.BaseDocument
        Document that has been parsed for dates. The results of
        the parsing are stored in the documents attrs. In particular,
        the attrs will contain a ``"date"`` key that will contain the
        parsed date information.
    """
    date_llm_caller = StructuredLLMCaller(
        llm_service=model_config.llm_service,
        usage_tracker=usage_tracker,
        **model_config.llm_call_kwargs,
    )
    doc.attrs["date"] = await DateExtractor(
        date_llm_caller, model_config.text_splitter
    ).parse(doc)

    return doc


async def extract_ordinance_text_with_llm(
    doc, text_splitter, extractor, original_text_key
):
    """Extract ordinance text from document using LLM

    Parameters
    ----------
    doc : elm.web.document.BaseDocument
        A document known to contain ordinance information. This means it
        must contain an ``"ordinance_text"`` key in the attrs. You can
        run :func:`~compass.extraction.apply.check_for_ordinance_info`
        to have this attribute populated automatically for documents
        that are found to contain ordinance data. Note that if the
        document's attrs does not contain the ``"ordinance_text"``
        key, you will get an error.
    text_splitter : obj
        Instance of an object that implements a `split_text` method.
        The method should take text as input (str) and return a list
        of text chunks. Langchain's text splitters should work for this
        input.
    extractor : compass.extraction.ordinance.WindOrdinanceTextExtractor
        Instance of
        :class:`~compass.extraction.ordinance.WindOrdinanceTextExtractor`
        used for ordinance text extraction.
    original_text_key : str
        String corresponding to the `doc.attrs` key containing the
        original text (before extraction).

    Returns
    -------
    elm.web.document.BaseDocument
        Document that has been parsed for ordinance text. The results of
        the extraction are stored in the document's attrs.
    str
        Key corresponding to the cleaned ordinance text stored in the
        `doc.attrs` dictionary.

    """
    prev_meta_name = original_text_key  # "ordinance_text"
    for meta_name, parser in extractor.parsers:
        doc.attrs[meta_name] = await _parse_if_input_text_not_empty(
            doc.attrs[prev_meta_name],
            text_splitter,
            parser,
            prev_meta_name,
            meta_name,
        )
        prev_meta_name = meta_name

    return doc, prev_meta_name


async def extract_ordinance_text_with_ngram_validation(
    doc,
    text_splitter,
    extractor,
    original_text_key,
    n=4,
    num_extraction_attempts=3,
    ngram_fraction_threshold=0.75,
):
    """Extract ordinance text for a single document with known ord info

    This extraction includes an "ngram" check, which attempts to detect
    whether or not the cleaned text was extracted from the original
    ordinance text. The processing will attempt to re-extract the text
    if the validation does not pass a certain threshold until the
    maximum number of attempts is reached. If the text still does not
    pass validation at this point, there is a good chance that the LLM
    hallucinated parts of the output text, so caution should be taken.

    Parameters
    ----------
    doc : elm.web.document.BaseDocument
        A document known to contain ordinance information. This means it
        must contain an ``"ordinance_text"`` key in the attrs. You can
        run :func:`~compass.extraction.apply.check_for_ordinance_info`
        to have this attribute populated automatically for documents
        that are found to contain ordinance data. Note that if the
        document's attrs does not contain the ``"ordinance_text"``
        key, it will not be processed.
    text_splitter : obj
        Instance of an object that implements a `split_text` method.
        The method should take text as input (str) and return a list
        of text chunks. Langchain's text splitters should work for this
        input.
    original_text_key : str
        String corresponding to the `doc.attrs` key containing the
        original text (before extraction).
    n : int, optional
        Number of words to include per ngram for the ngram validation,
        which helps ensure that the LLM did not hallucinate.
        By default, ``4``.
    num_extraction_attempts : int, optional
        Number of extraction attempts before returning text that did not
        pass the ngram check. If the processing exceeds this value,
        there is a good chance that the LLM hallucinated parts of the
        output text. Cannot be negative or 0. By default, ``3``.
    ngram_fraction_threshold : float, optional
        Fraction of ngrams in the cleaned text that are also found in
        the original ordinance text for the extraction to be considered
        successful. Should be a value between 0 and 1 (inclusive).
        By default, ``0.95``.

    Returns
    -------
    elm.web.document.BaseDocument
        Document that has been parsed for ordinance text. The results of
        the extraction are stored in the document's attrs.
    """
    if not doc.attrs.get(original_text_key):
        msg = (
            f"Input document has no {original_text_key!r} key or string "
            "does not contain information. Please run "
            "`check_for_ordinance_info` prior to calling this method."
        )
        warn(msg, COMPASSWarning)
        return doc

    return await _extract_with_ngram_check(
        doc,
        text_splitter,
        extractor,
        original_text_key,
        n=max(1, n),
        num_tries=max(1, num_extraction_attempts),
        ngram_fraction_threshold=max(0, min(1, ngram_fraction_threshold)),
    )


async def _extract_with_ngram_check(
    doc,
    text_splitter,
    extractor,
    original_text_key,
    n=4,
    num_tries=3,
    ngram_fraction_threshold=0.75,
):
    """Extract ordinance info from doc and validate using ngrams."""
    from compass.extraction.ngrams import sentence_ngram_containment  # noqa

    source = doc.attrs.get("source", "Unknown")
    original_text = doc.attrs[original_text_key]
    if not original_text:
        msg = (
            "Document missing original ordinance text! No extraction "
            "performed (Document source: %s)",
            source,
        )
        warn(msg, COMPASSWarning)
        return doc

    best_score = 0
    best_summary = ""
    out_text_key = "extracted_text"
    for attempt in range(1, num_tries + 1):
        doc, out_text_key = await extract_ordinance_text_with_llm(
            doc, text_splitter, extractor, original_text_key
        )
        cleaned_text = doc.attrs[out_text_key]
        if not cleaned_text:
            logger.debug(
                "No cleaned text found after extraction on attempt %d "
                "for document with source %s. Retrying...",
                attempt,
                source,
            )
            continue

        ngram_frac = sentence_ngram_containment(
            original=original_text, test=cleaned_text, n=n
        )
        if ngram_frac >= ngram_fraction_threshold:
            logger.debug(
                "Document extraction for %r passed ngram check on attempt %d "
                "with score %.2f (Document source: %s)",
                out_text_key,
                attempt + 1,
                ngram_frac,
                source,
            )
            best_score = ngram_frac
            break

        if ngram_frac > best_score:
            best_score = ngram_frac
            best_summary = cleaned_text

        logger.debug(
            "Document extraction for %r failed ngram check on attempt %d "
            "with score %.2f (Document source: %s). Retrying...",
            out_text_key,
            attempt + 1,
            ngram_frac,
            source,
        )
    else:
        doc.attrs[out_text_key] = best_summary
        msg = (
            f"Ngram check failed after {num_tries} tries. LLM hallucination "
            "in cleaned ordinance text is possible! Proceed with caution!! "
            f"(Score: {best_score:.2f}; Document source: {source})"
        )
        warn(msg, COMPASSWarning)

    doc.attrs["ngram_score"] = best_score
    return doc


async def extract_ordinance_values(doc, parser, text_key, out_key):
    """Extract ordinance values for a single document

    Document must be known to contain ordinance text.

    Parameters
    ----------
    doc : elm.web.document.BaseDocument
        A document known to contain ordinance text. This means it must
        contain an `text_key` key in the attrs. You can run
        :func:`~compass.extraction.apply.extract_ordinance_text_with_llm`
        to have this attribute populated automatically for documents
        that are found to contain ordinance data. Note that if the
        document's attrs does not contain the `text_key` key, it will
        not be processed.
    text_key : str
        Name of the key under which cleaned text is stored in
        `doc.attrs`. This text should be ready for extraction.
    out_key : str
        Name of the key under which extracted ordinances should be
        stored.

    Returns
    -------
    elm.web.document.BaseDocument
        Document that has been parsed for ordinance values. The results
        of the extraction are stored in the document's attrs.
    """
    if not doc.attrs.get(text_key):
        msg = (
            f"Input document has no {text_key!r} key or string "
            "does not contain info. Please run "
            "`extract_ordinance_text_with_llm` prior to calling this method."
        )
        warn(msg, COMPASSWarning)
        return doc

    doc.attrs[out_key] = await parser.parse(doc.attrs[text_key])
    return doc


async def _parse_if_input_text_not_empty(
    text, text_splitter, parser, curr_text_name, next_text_name
):
    """Extract text using parser, or return empty if input empty"""
    if not text:
        msg = (
            f"{curr_text_name!r} does not contain any text. Skipping "
            f"extraction for {next_text_name!r}"
        )
        warn(msg, COMPASSWarning)
        return text

    text_chunks = text_splitter.split_text(text)
    extracted_text = await parser(text_chunks)

    logger.debug_to_file(
        "Extracted text for %r is:\n%s", next_text_name, extracted_text
    )
    return extracted_text
