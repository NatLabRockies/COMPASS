"""COMPASS water rights extraction plugin"""

import logging
import importlib.resources
from pathlib import Path

import pandas as pd
from elm import EnergyWizard
from elm.embed import ChunkAndEmbed

from compass.extraction import extract_date
from compass.plugin import BaseExtractionPlugin, register_plugin
from compass.utilities.enums import LLMTasks
from compass.utilities.parsing import extract_ord_year_from_doc_attrs
from compass.exceptions import COMPASSRuntimeError
from compass.extraction.water.parse import StructuredWaterParser


logger = logging.getLogger(__name__)


WATER_RIGHTS_QUERY_TEMPLATES = [
    "{jurisdiction} rules",
    "{jurisdiction} management plan",
    "{jurisdiction} well permits",
    "{jurisdiction} well permit requirements",
    "requirements to drill a water well in {jurisdiction}",
]
BEST_WATER_RIGHTS_ORDINANCE_WEBSITE_URL_KEYWORDS = {
    "pdf": 92160,
    "water": 46080,
    "rights": 23040,
    "zoning": 11520,
    "ordinance": 5760,
    r"renewable%20energy": 1440,
    r"renewable+energy": 1440,
    "renewable energy": 1440,
    "planning": 720,
    "plan": 360,
    "government": 180,
    "code": 60,
    "area": 60,
    r"land%20development": 15,
    r"land+development": 15,
    "land development": 15,
    "land": 3,
    "environment": 3,
    "energy": 3,
    "renewable": 3,
    "municipal": 1,
    "department": 1,
}


class WaterRightsHeuristic:
    """NoOp heuristic check"""

    def check(self, *__, **___):  # noqa: PLR6301
        """Always return ``True`` for water rights documents"""
        return True


class TexasWaterRightsExtractor(BaseExtractionPlugin):
    """COMPASS solar extraction plugin"""

    IDENTIFIER = "tx water rights"
    """str: Identifier for extraction task """

    JURISDICTION_DATA_FP = (
        importlib.resources.files("compass")
        / "data"
        / "tx_water_districts.csv"
    )
    """:term:`path-like <path-like object>`: Path to Texas GCW names"""

    async def get_query_templates(self):  # noqa: PLR6301
        """Get a list of search engine query templates for extraction

        Query templates can contain the placeholder ``{jurisdiction}``
        which will be replaced with the full jurisdiction name during
        the search engine query.
        """
        return WATER_RIGHTS_QUERY_TEMPLATES

    async def get_website_keywords(self):  # noqa: PLR6301
        """Get a dict of website search keyword scores

        Dictionary mapping keywords to scores that indicate links which
        should be prioritized when performing a website scrape for a
        document.
        """
        return BEST_WATER_RIGHTS_ORDINANCE_WEBSITE_URL_KEYWORDS

    async def get_heuristic(self):  # noqa: PLR6301
        """Get a `BaseHeuristic` instance with a `check()` method

        The ``check()`` method should accept a string of text and return
        ``True`` if the text passes the heuristic check and ``False``
        otherwise.
        """
        return WaterRightsHeuristic()

    async def filter_docs(
        self,
        extraction_context,
        need_jurisdiction_verification=True,  # noqa: ARG002
    ):
        """Filter down candidate documents before parsing

        Parameters
        ----------
        extraction_context : ExtractionContext
            Context containing candidate documents to be filtered.
            Set the ``.documents`` attribute of this object to be the
            iterable of documents that should be kept for parsing.
        need_jurisdiction_verification : bool, optional
            Whether to verify that documents pertain to the correct
            jurisdiction. By default, ``True``.

        Returns
        -------
        ExtractionContext
            Context with filtered down documents.
        """
        model_config = self.model_configs.get(
            LLMTasks.EMBEDDING, self.model_configs[LLMTasks.DEFAULT]
        )
        _setup_endpoints(model_config)

        corpus = []
        for ind, doc in enumerate(extraction_context, start=1):
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
                embeddings = await obj.run_async(
                    rate_limit=model_config.llm_service_rate_limit
                )
                if any(e is None for e in embeddings):
                    msg = (
                        "Embeddings are ``None`` when building corpus for "
                        "water rights extraction!"
                    )
                    raise COMPASSRuntimeError(msg)  # noqa: TRY301

                corpus.append(
                    pd.DataFrame(
                        {
                            "text": obj.text_chunks.chunks,
                            "embedding": embeddings,
                        }
                    )
                )

            except Exception as e:  # noqa: BLE001
                logger.info("could not embed %r with error: %s", url, e)
                continue

            date_model_config = self.model_configs.get(
                LLMTasks.DATE_EXTRACTION, self.model_configs[LLMTasks.DEFAULT]
            )
            await extract_date(doc, date_model_config, self.usage_tracker)

            await extraction_context.mark_doc_as_data_source(
                doc, out_fn_stem=f"{self.jurisdiction.full_name} {ind}"
            )

        if len(corpus) == 0:
            logger.info(
                "No documents returned for %s, skipping",
                self.jurisdiction.full_name,
            )
            return None

        extraction_context.attrs["corpus"] = pd.concat(corpus)
        return extraction_context

    async def parse_docs_for_structured_data(self, extraction_context):
        """Parse documents to extract structured data/information

        Parameters
        ----------
        extraction_context : ExtractionContext
            Context containing candidate documents to parse.

        Returns
        -------
        ExtractionContext or None
            Context with extracted data/information stored in the
            ``.attrs`` dictionary, or ``None`` if no data was extracted.
        """
        model_config = self.model_configs.get(
            LLMTasks.DATA_EXTRACTION, self.model_configs[LLMTasks.DEFAULT]
        )

        logger.debug("Building energy wizard")
        wizard = EnergyWizard(
            extraction_context.attrs["corpus"],
            model=model_config.name,
        )

        logger.debug("Calling parser class")
        parser = StructuredWaterParser(
            wizard=wizard,
            location=self.jurisdiction.full_name,
            llm_service=model_config.llm_service,
            usage_tracker=self.usage_tracker,
            **model_config.llm_call_kwargs,
        )

        data_df = await parser.parse()
        data_df = _set_data_year(data_df, extraction_context)
        data_df = _set_data_sources(data_df, extraction_context)
        extraction_context.attrs["structured_data"] = data_df
        extraction_context.attrs["out_data_fn"] = (
            f"{self.jurisdiction.full_name} Water Rights.csv"
        )
        return extraction_context

    @classmethod
    def save_structured_data(cls, doc_infos, out_dir):
        """Write extracted water rights data to disk

        Parameters
        ----------
        doc_infos : list of dict
            List of dictionaries containing the following keys:

                - "jurisdiction": An initialized Jurisdiction object
                  representing the jurisdiction that was extracted.
                - "ord_db_fp": A path to the extracted structured data
                  stored on disk, or ``None`` if no data was extracted.

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


def _set_data_year(data_df, extraction_context):
    """Set the ordinance year column in the data DataFrame"""
    years = filter(
        None,
        [
            extract_ord_year_from_doc_attrs(doc.attrs)
            for doc in extraction_context
        ],
    )
    # TODO: is `max` the right one to use here?
    data_df["ord_year"] = max(years) if years else None
    return data_df


def _set_data_sources(data_df, extraction_context):
    """Set the source  column in the data DataFrame"""
    sources = filter(
        None, [doc.attrs.get("source") for doc in extraction_context]
    )
    data_df["source"] = " ;\n".join(sources) if sources else None
    return data_df


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


register_plugin(TexasWaterRightsExtractor)
