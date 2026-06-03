"""Top-level coordinator for the COMPASS pipeline"""

import json
import asyncio
import logging
from datetime import datetime, UTC
from abc import ABC, abstractmethod

from compass.services.openai import usage_from_response
from compass.services.usage import UsageTracker
from compass.exceptions import COMPASSError, COMPASSValueError
from compass.utilities import (
    compile_collection_summary_message,
    compile_run_summary_message,
    compute_total_cost_from_usage,
    load_all_jurisdiction_info,
    load_jurisdictions_from_fp,
    save_run_meta,
)
from compass.services.threaded import UsageUpdater
from compass.utilities.enums import COMPASSRunMode
from compass.utilities.jurisdictions import jurisdictions_from_df
from compass.utilities.logs import log_versions
from compass.utilities.parsing import convert_paths_to_strings
from compass.pipeline.collection.persistence import (
    build_collection_manifest,
    write_collection_manifest,
    load_collection_manifest,
)
from compass.pipeline import BaseRequest
from compass.pipeline.runtime import PipelineRuntime
from compass.pipeline.jurisdiction import SingleJurisdictionRun
from compass.pb import COMPASS_PB


logger = logging.getLogger(__name__)


async def run_compass(request):
    """Run the requested pipeline mode

    Parameters
    ----------
    request : compass.pipeline.data_classes.BaseRequest
        The request object containing all user-specified settings and
        configurations for the pipeline run. This should be an instance
        of one of the specific request types (e.g., ProcessRequest,
        CollectionRequest, ExtractionRequest) that inherit from
        BaseRequest, and should include all necessary information such
        as the mode to run in, output directories, jurisdiction
        information, model configurations, and any other relevant
        settings.

    Returns
    -------
    str
        A summary message of the pipeline run, including key information
        such as the number of jurisdictions processed, documents found,
        total cost, and output locations. The exact content of the
        message may vary depending on the mode that was run and the
        results of the processing.

    Raises
    ------
    COMPASSValueError
        If the request object is not of the expected type, or if it
        contains invalid configurations (e.g., no collection steps
        enabled in collection mode).
    """
    if not isinstance(request, BaseRequest):
        msg = "PipelineCoordinator.run expects a request object"
        raise COMPASSValueError(msg)

    if request.MODE == COMPASSRunMode.EXTRACT:
        steps = ["Extract collected documents"]
    else:
        steps = _enabled_steps(
            known_local_docs=request.known_sources.known_local_docs,
            known_doc_urls=request.known_sources.known_doc_urls,
            perform_se_search=request.perform_se_search,
            perform_website_search=request.perform_website_search,
        )

    runtime = PipelineRuntime(request)

    _log_execution_info(request, steps)
    jurisdictions_df = _load_jurisdictions_to_process(request.jurisdiction_fp)
    COMPASS_PB.create_main_task(
        num_jurisdictions=len(jurisdictions_df),
        action=request.MODE.pb_action_str,
    )
    async with runtime:
        try:
            return await _select_workflow(runtime).run(jurisdictions_df)
        except COMPASSError:
            raise
        except Exception:
            logger.exception("Fatal error during processing")
            raise


class BaseRunMode(ABC):
    """Strategy base class for mode-specific workflows"""

    def __init__(self, runtime):
        """

        Parameters
        ----------
        runtime : compass.pipeline.runtime.PipelineRuntime
            The runtime object containing all dependencies,
            configurations, and settings for the pipeline run. This
            object should be initialized with the user's request and any
            necessary setup (e.g., folder creation, model registry
            construction) before being passed to the workflow. The
            workflow will use the runtime to access configurations such
            as the mode to run in, the tech being processed, model
            configurations, known sources, and any other relevant
            settings needed to execute the workflow for the specified
            mode.

        """
        self.runtime = runtime

    def _create(
        self,
        jurisdiction,
        *,
        usage_tracker=None,
        validate_user_website_input=True,
    ):
        """Create one configured jurisdiction workflow"""
        extractor = self.runtime.extractor_class(
            jurisdiction=jurisdiction,
            model_configs=self.runtime.models,
            usage_tracker=usage_tracker,
        )
        return SingleJurisdictionRun(
            self.runtime,
            jurisdiction,
            extractor,
            usage_tracker=usage_tracker,
            known_local_docs=self.runtime.known_local_docs.get(
                jurisdiction.code
            ),
            known_doc_urls=self.runtime.known_doc_urls.get(jurisdiction.code),
            perform_se_search=self.runtime.request.perform_se_search,
            perform_website_search=(
                self.runtime.request.perform_website_search
            ),
            validate_user_website_input=validate_user_website_input,
        )

    @abstractmethod
    async def run(self, jurisdictions_df):
        """Run the mode workflow"""
        raise NotImplementedError


class COMPASSFullProcessing(BaseRunMode):
    """Concrete Strategy for full process mode"""

    async def run(self, jurisdictions_df):
        """Run process mode over all requested jurisdictions

        Parameters
        ----------
        jurisdictions_df : pandas.DataFrame
            A DataFrame containing information about the jurisdictions
            to process. This DataFrame should include all necessary
            information for each jurisdiction, such as its code, full
            name, and any other relevant metadata needed for processing.
            The workflow will iterate over each jurisdiction in the
            DataFrame and execute the full process pipeline (collection
            and extraction) for each one, using the information provided
            in the DataFrame to guide the processing steps.

        Returns
        -------
        str
            A summary message of the process run, including key
            information such as the number of jurisdictions processed,
            documents found, total cost, and output locations. The exact
            content of the message may vary depending on the results of
            the processing.
        """
        start_date = datetime.now(UTC)
        logger.info(
            "Processing %d jurisdiction(s) with continuous "
            "collection and extraction",
            len(jurisdictions_df),
        )
        tasks = []
        for jurisdiction in jurisdictions_from_df(jurisdictions_df):
            usage_tracker = UsageTracker(
                jurisdiction.full_name, usage_from_response
            )
            workflow = self._create(
                jurisdiction,
                usage_tracker=usage_tracker,
                validate_user_website_input=True,
            )
            tasks.append(
                asyncio.create_task(
                    workflow.run_process_with_logging(),
                    name=jurisdiction.full_name,
                )
            )

        results = await asyncio.gather(*tasks)
        return await _finalize_extraction(
            self.runtime, results, start_date, len(jurisdictions_df)
        )


class COMPASSCollection(BaseRunMode):
    """Concrete Strategy for document collection mode"""

    async def run(self, jurisdictions_df):
        """Run process mode over all requested jurisdictions

        Parameters
        ----------
        jurisdictions_df : pandas.DataFrame
            A DataFrame containing information about the jurisdictions
            to process. This DataFrame should include all necessary
            information for each jurisdiction, such as its code, full
            name, and any other relevant metadata needed for processing.
            The workflow will iterate over each jurisdiction in the
            DataFrame and execute the collection step for each one,
            using the information provided in the DataFrame to guide the
            processing steps.

        Returns
        -------
        str
            A summary message of the collection run, including key
            information such as the number of jurisdictions processed,
            documents found, total cost, and output locations. The exact
            content of the message may vary depending on the results of
            the processing.
        """
        logger.info(
            "Collecting documents for %d jurisdiction(s)",
            len(jurisdictions_df),
        )
        start_date = datetime.now(UTC)
        relative_to = (
            self.runtime.dirs.out
            if self.runtime.request.output_settings.make_paths_relative
            else None
        )
        tasks = []
        for jurisdiction in jurisdictions_from_df(jurisdictions_df):
            workflow = self._create(
                jurisdiction,
                usage_tracker=None,
                validate_user_website_input=False,
            )
            tasks.append(
                asyncio.create_task(
                    workflow.run_collection_with_logging(
                        relative_to=relative_to
                    ),
                    name=jurisdiction.full_name,
                )
            )

        collection_infos = await asyncio.gather(*tasks)
        manifest = build_collection_manifest(
            self.runtime.tech, collection_infos
        )
        manifest_fp = await write_collection_manifest(
            self.runtime.dirs.out, manifest
        )
        time_elapsed = datetime.now(UTC) - start_date
        collection_msg = compile_collection_summary_message(
            manifest_fp,
            manifest,
            total_seconds=time_elapsed.total_seconds(),
        )
        for sub_msg in collection_msg.split("\n"):
            logger.info(sub_msg)
        return collection_msg


class COMPASSExtraction(BaseRunMode):
    """Concrete Strategy for extraction mode over saved manifests"""

    async def run(self, jurisdictions_df):
        """Run process mode over all requested jurisdictions

        Parameters
        ----------
        jurisdictions_df : pandas.DataFrame
            A DataFrame containing information about the jurisdictions
            to process. This DataFrame should include all necessary
            information for each jurisdiction, such as its code, full
            name, and any other relevant metadata needed for processing.
            The workflow will iterate over each jurisdiction in the
            DataFrame and execute the extraction step for each one,
            using the information provided in the DataFrame to guide the
            processing steps.

        Returns
        -------
        str
            A summary message of the extraction run, including key
            information such as the number of jurisdictions processed,
            documents found, total cost, and output locations. The exact
            content of the message may vary depending on the results of
            the processing.
        """
        manifest = await load_collection_manifest(
            self.runtime.request.collection_manifest_fp, self.runtime.tech
        )
        jurisdictions = manifest.get("jurisdictions", [])
        logger.info(
            "Extracting structured data for %d jurisdiction(s)",
            len(jurisdictions),
        )

        tasks = []
        start_date = datetime.now(UTC)
        for jurisdiction in jurisdictions_from_df(jurisdictions_df):
            collection_info = [
                info
                for info in jurisdictions
                if info is not None and info.get("FIPS") == jurisdiction.code
            ]
            if not collection_info:
                logger.warning(
                    "No collection info found for %s; skipping extraction",
                    jurisdiction.full_name,
                )
                continue

            usage_tracker = UsageTracker(
                jurisdiction.full_name, usage_from_response
            )
            workflow = self._create(
                jurisdiction,
                usage_tracker=usage_tracker,
                validate_user_website_input=True,
            )
            tasks.append(
                asyncio.create_task(
                    workflow.run_extraction_with_logging(collection_info[0]),
                    name=jurisdiction.full_name,
                )
            )

        results = await asyncio.gather(*tasks)
        return await _finalize_extraction(
            self.runtime, results, start_date, len(jurisdictions_df)
        )


def _load_jurisdictions_to_process(jurisdiction_fp):
    """Load jurisdictions for the run"""
    if jurisdiction_fp is None:
        logger.info("No `jurisdiction_fp` input! Loading all jurisdictions")
        return load_all_jurisdiction_info()
    return load_jurisdictions_from_fp(jurisdiction_fp)


def _select_workflow(runtime):
    """Select the concrete mode workflow"""
    if runtime.mode == COMPASSRunMode.COLLECT:
        return COMPASSCollection(runtime)
    if runtime.mode == COMPASSRunMode.EXTRACT:
        return COMPASSExtraction(runtime)
    if runtime.mode == COMPASSRunMode.PROCESS:
        return COMPASSFullProcessing(runtime)

    msg = f"Unsupported mode: {runtime.mode}"
    raise COMPASSValueError(msg)


def _log_execution_info(request, steps):
    """Log execution metadata and normalized args"""
    log_versions(logger)
    logger.info(
        "Using the following document acquisition step(s):\n\t%s",
        " -> ".join(steps),
    )
    called_args = _request_to_log_args(request)
    normalized_args = convert_paths_to_strings(called_args)
    logger.debug_to_file(
        "Called process pipeline with:\n%s",
        json.dumps(normalized_args, indent=4),
    )


def _request_to_log_args(request):
    """Convert a request object into a loggable dictionary"""
    return {
        "mode": str(request.MODE),
        "tech": request.tech,
        "jurisdiction_fp": request.jurisdiction_fp,
        "collection_manifest_fp": request.collection_manifest_fp,
        "perform_se_search": request.perform_se_search,
        "perform_website_search": request.perform_website_search,
        "file_loader_kwargs": request.file_loader_kwargs,
        "search_settings": request.search_settings.__dict__,
        "runtime_settings": request.runtime_settings.__dict__,
        "output_settings": request.output_settings.__dict__,
        "known_sources": request.known_sources.__dict__,
        "model": request.user_model_input,
        "llm_costs": request.llm_costs,
    }


def _enabled_steps(
    known_local_docs=None,
    known_doc_urls=None,
    perform_se_search=True,
    perform_website_search=True,
):
    """Return enabled collection steps or raise when none are enabled"""
    steps = []
    if known_local_docs:
        steps.append("Check local document")
    if known_doc_urls:
        steps.append("Check known document URL")
    if perform_se_search:
        steps.append("Look for document using search engine")
    if perform_website_search:
        steps.append("Look for document on jurisdiction website")

    if not steps:
        msg = (
            "No processing steps enabled! Please provide at least one of "
            "'known_local_docs', 'known_doc_urls', or set at least one "
            "of 'perform_se_search' or 'perform_website_search' to True."
        )
        raise COMPASSValueError(msg)

    return steps


async def _finalize_extraction(
    runtime, results, start_date, num_jurisdictions
):
    """Finalize process or extraction mode outputs"""
    total_cost = await _compute_total_cost()
    doc_infos = [
        {
            "jurisdiction": result.jurisdiction,
            "ord_db_fp": result.ord_db_fp,
        }
        for result in results
        if result
    ]

    if doc_infos:
        num_docs_found = runtime.extractor_class.save_structured_data(
            doc_infos, runtime.dirs.out
        )
    else:
        num_docs_found = 0

    total_time = save_run_meta(
        runtime.dirs,
        runtime.tech,
        start_date=start_date,
        end_date=datetime.now(UTC),
        num_jurisdictions_searched=num_jurisdictions,
        num_jurisdictions_found=num_docs_found,
        total_cost=total_cost,
        models=runtime.models,
    )
    run_msg = compile_run_summary_message(
        total_seconds=total_time,
        total_cost=total_cost,
        out_dir=runtime.dirs.out,
        document_count=num_docs_found,
    )
    for sub_msg in run_msg.split("\n"):
        logger.info(sub_msg.replace("[#71906e]", "").replace("[/#71906e]", ""))
    return run_msg


async def _compute_total_cost():
    """Compute total cost from tracked usage"""
    total_usage = await UsageUpdater.call(None)
    if not total_usage:
        return 0
    return compute_total_cost_from_usage(total_usage)
