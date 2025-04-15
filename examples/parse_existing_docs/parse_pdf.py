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
from compass.extraction.apply import extract_ordinance_values
from compass.services.provider import RunningAsyncServices
from compass.extraction.apply import (
    check_for_ordinance_info,
    extract_ordinance_text_with_llm,
)
from compass.utilities.logs import AddLocationFilter


logger = logging.getLogger("compass")


def _setup_logging(log_level="INFO"):
    """Setup logging"""
    custom_theme = Theme({"logging.level.trace": "rgb(94,79,162)"})
    console = Console(theme=custom_theme)

    for lib in ["compass", "elm"]:
        logger = logging.getLogger(lib)
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


async def _extract_ordinances(doc, caller_args):
    """Run ordinance extraction pipeline"""

    async with RunningAsyncServices([caller_args.llm_service]):
        logger.info("Checking for ordinances in document...")
        doc = await check_for_ordinance_info(
            doc,
            model_config=caller_args,
            heuristic=SolarHeuristic(),
            ordinance_text_collector_class=SolarOrdinanceTextCollector,
            permitted_use_text_collector_class=None,
        )

        logger.info("Extracting ordinance text from document...")
        doc, ord_text_key = await extract_ordinance_text_with_llm(
            doc,
            caller_args.text_splitter,
            extractor=SolarOrdinanceTextExtractor(
                LLMCaller(llm_service=caller_args.llm_service)
            ),
            original_text_key="ordinance_text",
        )

        logger.info(
            "Extracting structured ordinance values from ordinance text..."
        )
        return await extract_ordinance_values(
            doc,
            parser=StructuredSolarOrdinanceParser(
                llm_service=caller_args.llm_service
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
    caller_args = OpenAIConfig(
        name="gpt-4o-mini",
        llm_call_kwargs={"temperature": 0},
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

    doc = asyncio.run(_extract_ordinances(doc, caller_args))

    # save outputs
    (
        doc.attrs["ordinance_values"]
        .drop(columns=["quantitative"], errors="ignore")
        .to_csv(fp_ord, index=False)
    )
    with Path(fp_txt_ord_text).open("w", encoding="utf-8") as fh:
        fh.write(doc.attrs["cleaned_ordinance_text"])
