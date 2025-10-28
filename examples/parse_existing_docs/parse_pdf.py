# noqa: INP001
"""Example on parsing an existing PDF file on-disk for ordinances"""

import logging
import asyncio
from pathlib import Path

from rich.theme import Theme
from rich.logging import RichHandler
from rich.console import Console

from elm.web.document import PDFDocument
from elm.utilities import validate_azure_api_params

from compass.llm import LLMCaller, OpenAIConfig
from compass.extraction.solar import (
    SolarOrdinanceTextExtractor,
    SolarHeuristic,
    SolarOrdinanceTextCollector,
    StructuredSolarOrdinanceParser,
)
from compass.services.provider import RunningAsyncServices
from compass.extraction.apply import (
    extract_ordinance_values,
    check_for_ordinance_info,
    extract_ordinance_text_with_llm,
)
from compass.utilities.logs import AddLocationFilter
from compass.utilities.enums import LLMTasks


logger = logging.getLogger("compass")


def _setup_logging(log_level="INFO"):
    """Setup logging"""
    custom_theme = Theme({"logging.level.trace": "rgb(94,79,162)"})
    console = Console(theme=custom_theme)

    handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        omit_repeated_times=True,
        markup=True,
    )
    fmt = logging.Formatter(
        fmt="[[magenta]%(location)s[/magenta]]: %(message)s",
        defaults={"location": "main"},
    )
    handler.setFormatter(fmt)
    handler.addFilter(AddLocationFilter())
    logger.addHandler(handler)
    logger.setLevel(log_level)


async def _extract_ordinances(doc, model_configs):
    """Run ordinance extraction pipeline"""

    services = [model.llm_service for model in set(model_configs.values())]
    async with RunningAsyncServices(services):
        logger.info("Checking for ordinances in document...")
        model_config = model_configs.get(
            LLMTasks.DOCUMENT_CONTENT_VALIDATION,
            model_configs[LLMTasks.DEFAULT],
        )
        doc = await check_for_ordinance_info(
            doc,
            model_config=model_config,
            heuristic=SolarHeuristic(),
            tech="solar",
            ordinance_text_collector_class=SolarOrdinanceTextCollector,
            permitted_use_text_collector_class=None,
        )

        logger.info("Extracting ordinance text from document...")
        model_config = model_configs.get(
            LLMTasks.ORDINANCE_TEXT_EXTRACTION,
            model_configs[LLMTasks.DEFAULT],
        )
        doc, ord_text_key = await extract_ordinance_text_with_llm(
            doc,
            model_config.text_splitter,
            extractor=SolarOrdinanceTextExtractor(
                LLMCaller(llm_service=model_config.llm_service)
            ),
            original_text_key="ordinance_text",
        )

        logger.info(
            "Extracting structured ordinance values from ordinance text..."
        )
        model_config = model_configs.get(
            LLMTasks.ORDINANCE_VALUE_EXTRACTION,
            model_configs[LLMTasks.DEFAULT],
        )
        return await extract_ordinance_values(
            doc,
            parser=StructuredSolarOrdinanceParser(
                llm_service=model_config.llm_service
            ),
            text_key=ord_text_key,
            out_key="ordinance_values",
        )


if __name__ == "__main__":
    _setup_logging(log_level="INFO")

    fp_pdf = "Decatur County, Indiana.pdf"
    fp_txt_ord_text = fp_pdf.replace(".pdf", " Ordinance Text.txt")
    fp_ord = fp_pdf.replace(".pdf", " Ordinances.csv")

    doc = PDFDocument.from_file(fp_pdf)

    # setup LLM calling parameters
    azure_api_key, azure_version, azure_endpoint = validate_azure_api_params()
    gpt_4o_mini_config = OpenAIConfig(
        name="gpt-4o-mini",
        llm_call_kwargs={"temperature": 0, "seed": 42, "timeout": 300},
        llm_service_rate_limit=500_000,
        text_splitter_chunk_size=10_000,
        text_splitter_chunk_overlap=500,
        client_type="azure",
        client_kwargs={
            "api_key": azure_api_key,
            "api_version": azure_version,
            "azure_endpoint": azure_endpoint,
        },
    )
    gpt_41_mini_config = OpenAIConfig(
        name="wetosa-gpt-4.1-mini",
        llm_call_kwargs={"temperature": 0, "seed": 42, "timeout": 300},
        llm_service_rate_limit=500_000,
        text_splitter_chunk_size=10_000,
        text_splitter_chunk_overlap=500,
        client_type="azure",
        client_kwargs={
            "api_key": azure_api_key,
            "api_version": azure_version,
            "azure_endpoint": azure_endpoint,
        },
    )
    gpt_41_config = OpenAIConfig(
        name="wetosa-gpt-4.1",
        llm_call_kwargs={"temperature": 0, "seed": 42, "timeout": 300},
        llm_service_rate_limit=500_000,
        text_splitter_chunk_size=10_000,
        text_splitter_chunk_overlap=500,
        client_type="azure",
        client_kwargs={
            "api_key": azure_api_key,
            "api_version": azure_version,
            "azure_endpoint": azure_endpoint,
        },
    )
    model_configs = {
        LLMTasks.DEFAULT: gpt_4o_mini_config,
        LLMTasks.DOCUMENT_CONTENT_VALIDATION: gpt_41_mini_config,
        LLMTasks.ORDINANCE_VALUE_EXTRACTION: gpt_41_mini_config,
        LLMTasks.ORDINANCE_TEXT_EXTRACTION: gpt_41_config,
    }
    doc = asyncio.run(_extract_ordinances(doc, model_configs))

    # save outputs
    (
        doc.attrs["ordinance_values"]
        .drop(columns=["quantitative"], errors="ignore")
        .to_csv(fp_ord, index=False)
    )
    with Path(fp_txt_ord_text).open("w", encoding="utf-8") as fh:
        fh.write(doc.attrs["cleaned_ordinance_text"])
