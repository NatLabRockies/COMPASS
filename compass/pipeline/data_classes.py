"""Data classes used for the COMPASS pipeline"""

from copy import deepcopy
import importlib.resources
from functools import cached_property

from elm.web.search.run import SEARCH_ENGINE_OPTIONS

from compass.llm import OpenAIConfig
from compass.services.usage import LLMRateTracker
from compass.utilities.enums import COMPASSRunMode, LLMTasks
from compass.utilities.io import load_config
from compass.exceptions import COMPASSValueError


_DOMAINS = load_config(
    importlib.resources.files("compass") / "data" / "domains.json5",
)


class RuntimeSettings:
    """Value Object for runtime and execution settings"""

    def __init__(
        self,
        td_kwargs=None,
        tpe_kwargs=None,
        ppe_kwargs=None,
        max_num_concurrent_jurisdictions=25,
        log_level="INFO",
        keep_async_logs=False,
    ):
        """

        Parameters
        ----------
        td_kwargs : dict, optional
            Additional keyword arguments to pass to
            :class:`tempfile.TemporaryDirectory`. The temporary
            directory is used to store documents which have not yet been
            confirmed to contain relevant information.
            By default, ``None``.
        tpe_kwargs : dict, optional
            Additional keyword arguments to pass to
            :class:`concurrent.futures.ThreadPoolExecutor`, used for
            I/O-bound tasks such as logging and file writes.
            By default, ``None``.
        ppe_kwargs : dict, optional
            Additional keyword arguments to pass to
            :class:`concurrent.futures.ProcessPoolExecutor`, used for
            CPU-bound tasks such as PDF loading and parsing.
            By default, ``None``.
        max_num_concurrent_jurisdictions : int, default=25
            Maximum number of jurisdictions to process concurrently.
            Limiting this can help manage memory usage when dealing with
            a large number of documents. By default, ``25``.
        log_level : str, default="INFO"
            Logging level for ordinance scraping and parsing (e.g.,
            "TRACE", "DEBUG", "INFO", "WARNING", or "ERROR").
            By default, ``"INFO"``.
        keep_async_logs : bool, default=False
            Option to store the full asynchronous log record to a file.
            This is only useful if you intend to monitor overall
            processing progress from a file instead of from the
            terminal. If ``True``, all of the unordered records are
            written to a "all.log" file in the `log_dir` directory.
            By default, ``False``.
        """
        self.td_kwargs = td_kwargs
        self.tpe_kwargs = tpe_kwargs
        self.ppe_kwargs = ppe_kwargs
        self.max_num_concurrent_jurisdictions = (
            max_num_concurrent_jurisdictions
        )
        self.log_level = log_level
        self.keep_async_logs = keep_async_logs


class OutputSettings:
    """Value Object for filesystem output settings"""

    def __init__(
        self,
        out_dir,
        log_dir=None,
        clean_dir=None,
        ordinance_file_dir=None,
        jurisdiction_dbs_dir=None,
        make_paths_relative=False,
    ):
        """

        Parameters
        ----------
        out_dir : path-like
            Path to the output directory. If it does not exist, it will
            be created. This directory will contain the saved collection
            manifest, downloaded ordinance documents, parsed document
            text, usage metadata, and default subdirectories for logs
            and intermediate outputs (unless otherwise specified).
        log_dir : path-like, optional
            Path to the directory for storing log files. If not
            provided, a ``logs`` subdirectory will be created inside
            `out_dir`. By default, ``None``.
        clean_dir : path-like, optional
            Path to the directory for storing cleaned ordinance text
            output. If not provided, a ``cleaned_text`` subdirectory
            will be created inside `out_dir`. By default, ``None``.
        ordinance_file_dir : path-like, optional
            Path to the directory where downloaded ordinance files (PDFs
            or HTML) for each jurisdiction are stored. If not provided,
            a ``ordinance_files`` subdirectory will be created inside
            `out_dir`. By default, ``None``.
        jurisdiction_dbs_dir : path-like, optional
            Path to the directory where parsed ordinance database files
            are stored for each jurisdiction. If not provided, a
            ``jurisdiction_dbs`` subdirectory will be created inside
            `out_dir`. By default, ``None``.
        make_paths_relative : bool, default=False
            Option to make all file paths in the saved collection
            manifest relative to the output directory. This can be
            helpful for sharing the manifest or for ensuring that it can
            be loaded correctly on a different machine. If ``False``,
            absolute paths are used in the manifest.
            By default, ``True``.
        """
        self.out_dir = out_dir
        self.log_dir = log_dir
        self.clean_dir = clean_dir
        self.ordinance_file_dir = ordinance_file_dir
        self.jurisdiction_dbs_dir = jurisdiction_dbs_dir
        self.make_paths_relative = make_paths_relative


class KnownSourcesInput:
    """Value Object for known documents and URL inputs"""

    def __init__(self, known_local_docs=None, known_doc_urls=None):
        """

        Parameters
        ----------
        known_local_docs : dict or path-like, optional
            A dictionary where keys are the jurisdiction codes (as
            strings) and values are lists of dictionaries containing
            information about each local document. Each document
            dictionary should contain at least the key ``"source_fp"``
            pointing to the full local document path. Additional keys
            are copied onto the loaded document as attributes. This
            input can also be a path to a JSON file containing the same
            mapping. By default, ``None``.
        known_doc_urls : dict or path-like, optional
            A dictionary where keys are the jurisdiction codes (as
            strings) and values are lists of dictionaries containing
            information about each known URL to check. Each document
            dictionary should contain at least the key ``"source"``
            representing the known document URL. Additional keys are
            copied onto the loaded document as attributes. This input
            can also be a path to a JSON file containing the same
            mapping. By default, ``None``.
        """
        self.known_local_docs = known_local_docs
        self.known_doc_urls = known_doc_urls


class WebSearchParams:
    """Capture configuration for jurisdiction web searches

    The class normalizes and stores search-related settings that are
    reused across multiple search operations, including browser
    concurrency, engine preferences, and filtering rules.

    Notes
    -----
    Instances lazily translate the provided search engine definitions
    into ELM-compatible keyword arguments via :attr:`se_kwargs`,
    enabling straightforward reuse when issuing queries.
    """

    def __init__(
        self,
        num_urls_to_check_per_jurisdiction=5,
        max_num_concurrent_browsers=10,
        max_num_concurrent_website_searches=None,
        website_crawl_timeout_seconds=3600,
        url_ignore_substrings=None,
        url_keep_substrings=None,
        search_engines=None,
        simple_se_result_sort=True,
        pytesseract_exe_fp=None,
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
        website_crawl_timeout_seconds : int, default=3600
            Maximum number of seconds to allow for a website crawl to
            complete before timing out. If the crawl exceeds this time,
            it will be terminated and documents accepted before the
            timeout will be returned. By default, ``3600``
        url_ignore_substrings : list of str, optional
            A list of substrings that, if found in any URL, will cause
            the URL to be excluded from search results and COMPASS
            website crawl candidates. This can be used to specify
            particular websites or entire domains to ignore. For
            example::

                url_ignore_substrings = [
                    "wikipedia",
                    "nlr.gov",
                    "www.co.delaware.in.us/documents/1649699794_0382.pdf",
                ]

            The above configuration would ignore all `wikipedia`
            articles, all websites on the NLR domain, and the specific
            file located at
            `www.co.delaware.in.us/documents/1649699794_0382.pdf`.
            This input will include all of the blacklisted domains from
            https://github.com/NatLabRockies/COMPASS/blob/main/compass/data/domains.json5,
            so you will need to whitelist any domains in that list that
            you want to allow. By default, ``None``.
        url_keep_substrings : list of str, optional
            A list of substrings that, if found in any URL, will cause
            the URL to be kept in search results and COMPASS website
            crawl candidates, regardless of the default blacklist or
            the `url_ignore_substrings` input. For example::

                url_keep_substrings = [
                    "my_ordinance_collection.edu",
                ]

            The above configuration would keep all URLs from
            "my_ordinance_collection.edu" despite the fact that ``.edu``
            urls are blacklisted by default. By default, ``None``.
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
        simple_se_result_sort : bool, default=True
            Flag indicating whether to use a simple top-n sort from the
            first search engine that gives results (``True``) or to
            apply a holistic link sorting based on all results from all
            search engines (``False``). By default, ``True``.
        pytesseract_exe_fp : path-like, optional
            Path to the `pytesseract` executable. If specified, OCR will
            be used to extract text from scanned PDFs using Google's
            Tesseract. By default ``None``.
        """
        self.num_urls_to_check_per_jurisdiction = (
            num_urls_to_check_per_jurisdiction
        )
        self.max_num_concurrent_browsers = max_num_concurrent_browsers
        self.max_num_concurrent_website_searches = (
            max_num_concurrent_website_searches
        )
        self.website_crawl_timeout_seconds = website_crawl_timeout_seconds
        self.url_ignore_substrings = [
            *_DOMAINS["blacklist"],
            *(url_ignore_substrings or []),
        ]
        self.url_keep_substrings = [
            *_DOMAINS["whitelist"],
            *(url_keep_substrings or []),
        ]
        self._search_engines_input = search_engines
        self.simple_se_result_sort = simple_se_result_sort
        self.pytesseract_exe_fp = pytesseract_exe_fp

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


class DocParsingParams:
    """Value Object for document parsing settings"""

    def __init__(self, max_num_docs_per_jurisdiction=None):
        """

        Parameters
        ----------
        max_num_docs_per_jurisdiction : int, optional
            Maximum number of documents to parse for each jurisdiction
            (regardless of the collection method). If ``None``, all
            collected documents are parsed. By default, ``None``.
        """
        self.max_num_docs_per_jurisdiction = max_num_docs_per_jurisdiction


class BaseRequest:
    """Parameter Object base class for pipeline requests"""

    MODE = None
    """COMPASSRunMode associated with this request type"""

    def __init__(  # ruff:ignore[too-many-arguments]
        self,
        out_dir,
        tech,
        jurisdiction_fp,
        *,
        model="gpt-4o-mini",
        llm_costs=None,
        num_urls_to_check_per_jurisdiction=5,
        max_num_docs_to_parse_per_jurisdiction=None,
        max_num_concurrent_browsers=10,
        max_num_concurrent_website_searches=10,
        max_num_concurrent_jurisdictions=25,
        website_crawl_timeout_seconds=3600,
        url_ignore_substrings=None,
        url_keep_substrings=None,
        known_local_docs=None,
        known_doc_urls=None,
        file_loader_kwargs=None,
        search_engines=None,
        simple_se_result_sort=True,
        pytesseract_exe_fp=None,
        td_kwargs=None,
        tpe_kwargs=None,
        ppe_kwargs=None,
        log_dir=None,
        clean_dir=None,
        ordinance_file_dir=None,
        jurisdiction_dbs_dir=None,
        perform_se_search=True,
        perform_website_search=True,
        make_paths_relative=False,
        log_level="INFO",
        keep_async_logs=False,
        collection_manifest_fp=None,
    ):
        """

        Parameters
        ----------
        out_dir : path-like
            Path to the output directory. If it does not exist, it will
            be created. This directory will contain the saved collection
            manifest, downloaded ordinance documents, parsed document
            text, usage metadata, and default subdirectories for logs
            and intermediate outputs (unless otherwise specified).
        tech : str
            Label indicating which technology type is being processed.
            Must be one of the keys of
            :obj:`~compass.plugin.registry.PLUGIN_REGISTRY`.
        jurisdiction_fp : path-like
            Path to a CSV file specifying the jurisdictions to process.
            The CSV must contain at least two columns: "County" and
            "State", which specify the county and state names,
            respectively. If you would like to process a subdivision
            with a county, you must also include "Subdivision" and
            "Jurisdiction Type" columns. The "Subdivision" should be the
            name of the subdivision, and the "Jurisdiction Type" should
            be a string identifying the type of subdivision (e.g.,
            "City", "Township", etc.)
        model : str or list of dict, default="gpt-4o-mini"
            LLM model(s) to use for scraping and parsing ordinance
            documents. If a string is provided, it is assumed to be the
            name of the default model (e.g., "gpt-4o"), and environment
            variables are used for authentication.

            If a list is provided, it should contain dictionaries of
            arguments that can initialize instances of
            :class:`~compass.llm.config.OpenAIConfig`. Each dictionary
            can specify the model name, client type, and initialization
            arguments.

            Each dictionary must also include a ``tasks`` key, which
            maps to a string or list of strings indicating the tasks
            that instance should handle. Exactly one of the instances
            **must** include "default" as a task, which will be used
            when no specific task is matched. For example::

                "model": [
                    {
                        "model": "gpt-4o-mini",
                        "llm_call_kwargs": {
                            "temperature": 0,
                            "timeout": 300,
                        },
                        "client_kwargs": {
                            "api_key": "<your_api_key>",
                            "api_version": "<your_api_version>",
                            "azure_endpoint": "<your_azure_endpoint>",
                        },
                        "tasks": ["default", "date_extraction"],
                    },
                    {
                        "model": "gpt-4o",
                        "client_type": "openai",
                        "tasks": ["ordinance_text_extraction"],
                    }
                ]

            .. IMPORTANT::
                You will need to ensure that the model name used here
                matches your deployment if you are using Azure OpenAI.
                For example, if you deployed the GPT-4o-mini model under
                the name ``"gpt-4o-mini-2025-04-11"``, you would want to
                set ``"model": "gpt-4o-mini-2025-04-11"``.

            By default, ``"gpt-4o-mini"``.
        llm_costs : dict, optional
            Dictionary mapping model names to their token costs, used to
            track the estimated total cost of LLM usage during the run.
            The structure should be::

                {"model_name": {"prompt": float, "response": float}}

            Costs are specified in dollars per million tokens.
            For example::

                "llm_costs": {"my_gpt": {"prompt": 1.5, "response": 3}}

            registers a model named `"my_gpt"` with a cost of $1.5 per
            million input (prompt) tokens and $3 per million output
            (response) tokens for the current processing run.

            .. NOTE::

                The displayed total cost does not track cached tokens,
                so treat it like an estimate. Your final API costs may
                vary.

            If set to ``None``, no custom model costs are recorded, and
            cost tracking may be unavailable in the progress bar.
            By default, ``None``.
        num_urls_to_check_per_jurisdiction : int, default=5
            Number of unique Google search result URLs to check for each
            jurisdiction when attempting to locate ordinance documents.
            By default, ``5``.
        max_num_docs_to_parse_per_jurisdiction : int, optional
            Maximum number of documents to parse for each jurisdiction
            (regardless of the collection method). If ``None``, all
            collected documents are parsed. By default, ``None``.
        max_num_concurrent_browsers : int, default=10
            Maximum number of browser instances to launch concurrently
            for retrieving information from the web. Increasing this
            value too much may lead to timeouts or performance issues on
            machines with limited resources. By default, ``10``.
        max_num_concurrent_website_searches : int, default=10
            Maximum number of website searches allowed to run
            simultaneously. Increasing this value can speed up searches,
            but may lead to timeouts or performance issues on machines
            with limited resources. By default, ``10``.
        max_num_concurrent_jurisdictions : int, default=25
            Maximum number of jurisdictions to process concurrently.
            Limiting this can help manage memory usage when dealing with
            a large number of documents. By default, ``25``.
        website_crawl_timeout_seconds : int, default=3600
            Maximum number of seconds to allow for a website crawl to
            complete before timing out. If the crawl exceeds this time,
            it will be terminated and documents accepted before the
            timeout will be returned. By default, ``3600``
        url_ignore_substrings : list of str, optional
            A list of substrings that, if found in any URL, will cause
            the URL to be excluded from search results and COMPASS
            website crawl candidates. This can be used to specify
            particular websites or entire domains to ignore. For
            example::

                url_ignore_substrings = [
                    "wikipedia",
                    "nlr.gov",
                    "www.co.delaware.in.us/documents/1649699794_0382.pdf",
                ]

            The above configuration would ignore all `wikipedia`
            articles, all websites on the NLR domain, and the specific
            file located at
            `www.co.delaware.in.us/documents/1649699794_0382.pdf`.
            This input will include all of the blacklisted domains from
            https://github.com/NatLabRockies/COMPASS/blob/main/compass/data/domains.json5,
            so you will need to whitelist any domains in that list that
            you want to allow. By default, ``None``.
        url_keep_substrings : list of str, optional
            A list of substrings that, if found in any URL, will cause
            the URL to be kept in search results and COMPASS website
            crawl candidates, regardless of the default blacklist or
            the `url_ignore_substrings` input. For example::

                url_keep_substrings = [
                    "my_ordinance_collection.edu",
                ]

            The above configuration would keep all URLs from
            "my_ordinance_collection.edu" despite the fact that ``.edu``
            urls are blacklisted by default. By default, ``None``.
        known_local_docs : dict or path-like, optional
            A dictionary where keys are the jurisdiction codes (as
            strings) and values are lists of dictionaries containing
            information about each local document. Each document
            dictionary should contain at least the key ``"source_fp"``
            pointing to the full local document path. Additional keys
            are copied onto the loaded document as attributes. This
            input can also be a path to a JSON file containing the same
            mapping. By default, ``None``.
        known_doc_urls : dict or path-like, optional
            A dictionary where keys are the jurisdiction codes (as
            strings) and values are lists of dictionaries containing
            information about each known URL to check. Each document
            dictionary should contain at least the key ``"source"``
            representing the known document URL. Additional keys are
            copied onto the loaded document as attributes. This input
            can also be a path to a JSON file containing the same
            mapping. By default, ``None``.
        file_loader_kwargs : dict, optional
            Dictionary of keyword argument pairs to initialize
            :class:`elm.web.file_loader.AsyncWebFileLoader`. If found,
            the ``"pw_launch_kwargs"`` key in these will also be used to
            initialize the Playwright-backed Google search used for
            search engine retrieval. By default, ``None``.
        search_engines : list, optional
            A list of dictionaries describing the search engine classes
            and keyword arguments to use for search engine retrieval. If
            ``None``, the default search engine configurations and
            fallback order are used. By default, ``None``.
        simple_se_result_sort : bool, default=True
            Flag indicating whether to use a simple top-n sort from the
            first search engine that gives results (``True``) or to
            apply a holistic link sorting based on all results from all
            search engines (``False``). By default, ``True``.
        pytesseract_exe_fp : path-like, optional
            Path to the `pytesseract` executable. If specified, OCR will
            be used to extract text from scanned PDFs using Google's
            Tesseract. By default, ``None``.
        td_kwargs : dict, optional
            Additional keyword arguments to pass to
            :class:`tempfile.TemporaryDirectory`. The temporary
            directory is used to store documents which have not yet been
            confirmed to contain relevant information.
            By default, ``None``.
        tpe_kwargs : dict, optional
            Additional keyword arguments to pass to
            :class:`concurrent.futures.ThreadPoolExecutor`, used for
            I/O-bound tasks such as logging and file writes.
            By default, ``None``.
        ppe_kwargs : dict, optional
            Additional keyword arguments to pass to
            :class:`concurrent.futures.ProcessPoolExecutor`, used for
            CPU-bound tasks such as PDF loading and parsing.
            By default, ``None``.
        log_dir : path-like, optional
            Path to the directory for storing log files. If not
            provided, a ``logs`` subdirectory will be created inside
            `out_dir`. By default, ``None``.
        clean_dir : path-like, optional
            Path to the directory for storing cleaned ordinance text
            output. If not provided, a ``cleaned_text`` subdirectory
            will be created inside `out_dir`. By default, ``None``.
        ordinance_file_dir : path-like, optional
            Path to the directory where downloaded ordinance files (PDFs
            or HTML) for each jurisdiction are stored. If not provided,
            a ``ordinance_files`` subdirectory will be created inside
            `out_dir`. By default, ``None``.
        jurisdiction_dbs_dir : path-like, optional
            Path to the directory where parsed ordinance database files
            are stored for each jurisdiction. If not provided, a
            ``jurisdiction_dbs`` subdirectory will be created inside
            `out_dir`. By default, ``None``.
        perform_se_search : bool, default=True
            Option to perform a search engine-based search for ordinance
            documents. This is the standard way to collect ordinance
            documents, and it is recommended to leave this set to
            ``True`` unless you are re-processing local documents. If
            ``True``, the search engine approach is used to locate
            ordinance documents
            before falling back to a website crawl-based search (if that
            has been selected). By default, ``True``.
        perform_website_search : bool, default=True
            Option to fallback to a jurisdiction website crawl-based
            search for ordinance documents if the search engine approach
            fails to recover any relevant documents.
            By default, ``True``.
        make_paths_relative : bool, default=False
            Option to make all file paths in the saved collection
            manifest relative to the output directory. This can be
            helpful for sharing the manifest or for ensuring that it can
            be loaded correctly on a different machine. If ``False``,
            absolute paths are used in the manifest.
            By default, ``False``.
        log_level : str, default="INFO"
            Logging level for ordinance scraping and parsing (e.g.,
            "TRACE", "DEBUG", "INFO", "WARNING", or "ERROR").
            By default, ``"INFO"``.
        keep_async_logs : bool, default=False
            Option to store the full asynchronous log record to a file.
            This is only useful if you intend to monitor overall
            processing progress from a file instead of from the
            terminal. If ``True``, all of the unordered records are
            written to a "all.log" file in the `log_dir` directory.
            By default, ``False``.
        collection_manifest_fp : path-like or list of path-like, optional
            Path to the JSON collection manifest created by the document
            collection step. This can be a single path or a list of
            paths for multiple collection manifests, any of which may
            include glob patterns. Each collection manifest must contain
            the persisted document information needed to reload each
            collected document for extraction. Only needed if running in
            extraction mode with a separate collection step.
            By default, ``None``.
        """  # ruff:ignore[doc-line-too-long]
        self.tech = tech
        self.jurisdiction_fp = jurisdiction_fp
        self.perform_se_search = perform_se_search
        self.perform_website_search = perform_website_search
        self.collection_manifest_fp = collection_manifest_fp
        self.file_loader_kwargs = file_loader_kwargs

        self.search_settings = WebSearchParams(
            num_urls_to_check_per_jurisdiction=(
                num_urls_to_check_per_jurisdiction
            ),
            max_num_concurrent_browsers=max_num_concurrent_browsers,
            max_num_concurrent_website_searches=(
                max_num_concurrent_website_searches
            ),
            website_crawl_timeout_seconds=website_crawl_timeout_seconds,
            url_ignore_substrings=url_ignore_substrings,
            url_keep_substrings=url_keep_substrings,
            search_engines=search_engines,
            simple_se_result_sort=simple_se_result_sort,
            pytesseract_exe_fp=pytesseract_exe_fp,
        )
        self.parsing_settings = DocParsingParams(
            max_num_docs_per_jurisdiction=(
                max_num_docs_to_parse_per_jurisdiction
            )
        )
        self.runtime_settings = RuntimeSettings(
            td_kwargs=td_kwargs,
            tpe_kwargs=tpe_kwargs,
            ppe_kwargs=ppe_kwargs,
            max_num_concurrent_jurisdictions=(
                max_num_concurrent_jurisdictions
            ),
            log_level=log_level,
            keep_async_logs=keep_async_logs,
        )
        self.output_settings = OutputSettings(
            out_dir=out_dir,
            log_dir=log_dir,
            clean_dir=clean_dir,
            ordinance_file_dir=ordinance_file_dir,
            jurisdiction_dbs_dir=jurisdiction_dbs_dir,
            make_paths_relative=make_paths_relative,
        )
        self.known_sources = KnownSourcesInput(
            known_local_docs=known_local_docs,
            known_doc_urls=known_doc_urls,
        )
        self.user_model_input = model
        self.llm_costs = llm_costs
        self._rate_tracker = LLMRateTracker()

    @cached_property
    def models(self):
        """dict: Mapping of LLM task to OpenAIConfig for this request"""
        if not self.user_model_input:
            return {}

        return build_models(
            self.user_model_input, rate_tracker=self._rate_tracker
        )

    @property
    def rate_tracker(self):
        """LLMRateTracker: Rate tracker for LLM calls"""
        return self._rate_tracker


class ProcessRequest(BaseRequest):
    """Parameter Object for full process mode"""

    MODE = COMPASSRunMode.PROCESS
    """COMPASSRunMode associated with this request type"""


class CollectionRequest(BaseRequest):
    """Parameter Object for collection mode"""

    MODE = COMPASSRunMode.COLLECT
    """COMPASSRunMode associated with this request type"""

    def __init__(  # ruff:ignore[too-many-arguments]
        self,
        out_dir,
        tech,
        jurisdiction_fp,
        *,
        model=None,
        num_urls_to_check_per_jurisdiction=5,
        max_num_concurrent_browsers=10,
        max_num_concurrent_website_searches=10,
        max_num_concurrent_jurisdictions=25,
        website_crawl_timeout_seconds=3600,
        url_ignore_substrings=None,
        url_keep_substrings=None,
        known_local_docs=None,
        known_doc_urls=None,
        file_loader_kwargs=None,
        search_engines=None,
        simple_se_result_sort=True,
        pytesseract_exe_fp=None,
        td_kwargs=None,
        tpe_kwargs=None,
        ppe_kwargs=None,
        log_dir=None,
        source_file_dir=None,
        parsed_file_dir=None,
        shard_dir=None,
        perform_se_search=True,
        perform_website_search=True,
        make_paths_relative=True,
        llm_costs=None,
        log_level="INFO",
        keep_async_logs=False,
    ):
        """

        Parameters
        ----------
        out_dir : path-like
            Path to the output directory. If it does not exist, it will
            be created. This directory will contain the saved collection
            manifest, downloaded ordinance documents, parsed document
            text, usage metadata, and default subdirectories for logs
            and intermediate outputs (unless otherwise specified).
        tech : str
            Label indicating which technology type is being processed.
            Must be one of the keys of
            :obj:`~compass.plugin.registry.PLUGIN_REGISTRY`.
        jurisdiction_fp : path-like
            Path to a CSV file specifying the jurisdictions to process.
            The CSV must contain at least two columns: "County" and
            "State", which specify the county and state names,
            respectively. If you would like to process a subdivision
            with a county, you must also include "Subdivision" and
            "Jurisdiction Type" columns. The "Subdivision" should be the
            name of the subdivision, and the "Jurisdiction Type" should
            be a string identifying the type of subdivision (e.g.,
            "City", "Township", etc.)
        model : str or list of dict, optional
            Optional model configuration used only for collection-side
            LLM tasks, such as:

                - Searching for and validating a jurisdiction website
                  before website crawl

            If this key is left out, these steps are skipped completely.
            If provided as a string, it is assumed to be the name of the
            default model (e.g., "gpt-5-mini"), and environment
            variables are used for authentication.

            If a list is provided, it should contain dictionaries of
            arguments that can initialize instances of
            :class:`~compass.llm.config.OpenAIConfig`. Each dictionary
            can specify the model name, client type, and initialization
            arguments.

            Each dictionary must also include a ``tasks`` key, which
            maps to a string or list of strings indicating the tasks
            that instance should handle. Exactly one of the instances
            **must** include "default" as a task, which will be used
            when no specific task is matched. For example::

                "model": [
                    {
                        "model": "gpt-4o-mini",
                        "llm_call_kwargs": {
                            "temperature": 0,
                            "timeout": 300,
                        },
                        "client_kwargs": {
                            "api_key": "<your_api_key>",
                            "api_version": "<your_api_version>",
                            "azure_endpoint": "<your_azure_endpoint>",
                        },
                        "tasks": ["default", "date_extraction"],
                    },
                    {
                        "model": "gpt-4o",
                        "client_type": "openai",
                        "tasks": ["ordinance_text_extraction"],
                    }
                ]

            .. IMPORTANT::
                You will need to ensure that the model name used here
                matches your deployment if you are using Azure OpenAI.
                For example, if you deployed the GPT-4o-mini model under
                the name ``"gpt-4o-mini-2025-04-11"``, you would want to
                set ``"model": "gpt-4o-mini-2025-04-11"``.

            By default, ``None``.
        num_urls_to_check_per_jurisdiction : int, default=5
            Number of unique Google search result URLs to check for each
            jurisdiction when attempting to locate ordinance documents.
            By default, ``5``.
        max_num_concurrent_browsers : int, default=10
            Maximum number of browser instances to launch concurrently
            for retrieving information from the web. Increasing this
            value too much may lead to timeouts or performance issues on
            machines with limited resources. By default, ``10``.
        max_num_concurrent_website_searches : int, default=10
            Maximum number of website searches allowed to run
            simultaneously. Increasing this value can speed up searches,
            but may lead to timeouts or performance issues on machines
            with limited resources. By default, ``10``.
        max_num_concurrent_jurisdictions : int, default=25
            Maximum number of jurisdictions to process concurrently.
            Limiting this can help manage memory usage when dealing with
            a large number of documents. By default, ``25``.
        website_crawl_timeout_seconds : int, default=3600
            Maximum number of seconds to allow for a website crawl to
            complete before timing out. If the crawl exceeds this time,
            it will be terminated and documents accepted before the
            timeout will be returned. By default, ``3600``
        url_ignore_substrings : list of str, optional
            A list of substrings that, if found in any URL, will cause
            the URL to be excluded from search results and COMPASS
            website crawl candidates. This can be used to specify
            particular websites or entire domains to ignore. For
            example::

                url_ignore_substrings = [
                    "wikipedia",
                    "nlr.gov",
                    "www.co.delaware.in.us/documents/1649699794_0382.pdf",
                ]

            The above configuration would ignore all `wikipedia`
            articles, all websites on the NLR domain, and the specific
            file located at
            `www.co.delaware.in.us/documents/1649699794_0382.pdf`.
            This input will include all of the blacklisted domains from
            https://github.com/NatLabRockies/COMPASS/blob/main/compass/data/domains.json5,
            so you will need to whitelist any domains in that list that
            you want to allow. By default, ``None``.
        url_keep_substrings : list of str, optional
            A list of substrings that, if found in any URL, will cause
            the URL to be kept in search results and COMPASS website
            crawl candidates, regardless of the default blacklist or
            the `url_ignore_substrings` input. For example::

                url_keep_substrings = [
                    "my_ordinance_collection.edu",
                ]

            The above configuration would keep all URLs from
            "my_ordinance_collection.edu" despite the fact that ``.edu``
            urls are blacklisted by default. By default, ``None``.
        known_local_docs : dict or path-like, optional
            A dictionary where keys are the jurisdiction codes (as
            strings) and values are lists of dictionaries containing
            information about each local document. Each document
            dictionary should contain at least the key ``"source_fp"``
            pointing to the full local document path. Additional keys
            are copied onto the loaded document as attributes. This
            input can also be a path to a JSON file containing the same
            mapping. By default, ``None``.
        known_doc_urls : dict or path-like, optional
            A dictionary where keys are the jurisdiction codes (as
            strings) and values are lists of dictionaries containing
            information about each known URL to check. Each document
            dictionary should contain at least the key ``"source"``
            representing the known document URL. Additional keys are
            copied onto the loaded document as attributes. This input
            can also be a path to a JSON file containing the same
            mapping. By default, ``None``.
        file_loader_kwargs : dict, optional
            Dictionary of keyword argument pairs to initialize
            :class:`elm.web.file_loader.AsyncWebFileLoader`. If found,
            the ``"pw_launch_kwargs"`` key in these will also be used to
            initialize the Playwright-backed Google search used for
            search engine retrieval. By default, ``None``.
        search_engines : list, optional
            A list of dictionaries describing the search engine classes
            and keyword arguments to use for search engine retrieval. If
            ``None``, the default search engine configurations and
            fallback order are used. By default, ``None``.
        simple_se_result_sort : bool, default=True
            Flag indicating whether to use a simple top-n sort from the
            first search engine that gives results (``True``) or to
            apply a holistic link sorting based on all results from all
            search engines (``False``). By default, ``True``.
        pytesseract_exe_fp : path-like, optional
            Path to the `pytesseract` executable. If specified, OCR will
            be used to extract text from scanned PDFs using Google's
            Tesseract. By default, ``None``.
        td_kwargs : dict, optional
            Additional keyword arguments to pass to
            :class:`tempfile.TemporaryDirectory`. The temporary
            directory is used to store documents which have not yet been
            confirmed to contain relevant information.
            By default, ``None``.
        tpe_kwargs : dict, optional
            Additional keyword arguments to pass to
            :class:`concurrent.futures.ThreadPoolExecutor`, used for
            I/O-bound tasks such as logging and file writes.
            By default, ``None``.
        ppe_kwargs : dict, optional
            Additional keyword arguments to pass to
            :class:`concurrent.futures.ProcessPoolExecutor`, used for
            CPU-bound tasks such as PDF loading and parsing.
            By default, ``None``.
        log_dir : path-like, optional
            Path to the directory for storing log files. If not
            provided, a ``logs`` subdirectory will be created inside
            `out_dir`. By default, ``None``.
        source_file_dir : path-like, optional
            Path to the directory where collected source ordinance files
            (PDFs or HTML) are stored. If not provided, an
            ``ordinance_files`` subdirectory will be created inside
            `out_dir`. By default, ``None``.
        parsed_file_dir : path-like, optional
            Path to the directory where parsed document text files are
            stored. If not provided, a ``cleaned_text`` subdirectory
            will be created inside `out_dir`. By default, ``None``.
        shard_dir : path-like, optional
            Path to the directory for storing per-jurisdiction
            collection manifest shards. If not provided, a
            ``manifest_shards`` subdirectory will be created inside
            `out_dir`. By default, ``None``.
        perform_se_search : bool, default=True
            Option to perform a search engine-based search for ordinance
            documents. This is the standard way to collect ordinance
            documents, and it is recommended to leave this set to
            ``True`` unless you are re-processing local documents. If
            ``True``, the search engine approach is used to locate
            ordinance documents
            before falling back to a website crawl-based search (if that
            has been selected). By default, ``True``.
        perform_website_search : bool, default=True
            Option to fallback to a jurisdiction website crawl-based
            search for ordinance documents if the search engine approach
            fails to recover any relevant documents.
            By default, ``True``.
        make_paths_relative : bool, default=True
            Option to make all file paths in the saved collection
            manifest relative to the output directory. This can be
            helpful for sharing the manifest or for ensuring that it can
            be loaded correctly on a different machine. If ``False``,
            absolute paths are used in the manifest.
            By default, ``True``.
        llm_costs : dict, optional
            Dictionary mapping model names to their token costs, used to
            track the estimated total cost of LLM usage during the run.
            The structure should be::

                {"model_name": {"prompt": float, "response": float}}

            Costs are specified in dollars per million tokens.
            For example::

                "llm_costs": {"my_gpt": {"prompt": 1.5, "response": 3}}

            registers a model named `"my_gpt"` with a cost of $1.5 per
            million input (prompt) tokens and $3 per million output
            (response) tokens for the current processing run.

            .. NOTE::

                The displayed total cost does not track cached tokens,
                so treat it like an estimate. Your final API costs may
                vary.

            If set to ``None``, no custom model costs are recorded, and
            cost tracking may be unavailable in the progress bar.
            By default, ``None``.
        log_level : str, default="INFO"
            Logging level for ordinance scraping and parsing (e.g.,
            "TRACE", "DEBUG", "INFO", "WARNING", or "ERROR").
            By default, ``"INFO"``.
        keep_async_logs : bool, default=False
            Option to store the full asynchronous log record to a file.
            This is only useful if you intend to monitor overall
            processing progress from a file instead of from the
            terminal. If ``True``, all of the unordered records are
            written to a "all.log" file in the `log_dir` directory.
            By default, ``False``.
        """
        super().__init__(
            out_dir=out_dir,
            tech=tech,
            jurisdiction_fp=jurisdiction_fp,
            model=model,
            llm_costs=llm_costs,
            num_urls_to_check_per_jurisdiction=(
                num_urls_to_check_per_jurisdiction
            ),
            max_num_concurrent_browsers=max_num_concurrent_browsers,
            max_num_concurrent_website_searches=(
                max_num_concurrent_website_searches
            ),
            max_num_concurrent_jurisdictions=max_num_concurrent_jurisdictions,
            website_crawl_timeout_seconds=website_crawl_timeout_seconds,
            url_ignore_substrings=url_ignore_substrings,
            url_keep_substrings=url_keep_substrings,
            known_local_docs=known_local_docs,
            known_doc_urls=known_doc_urls,
            file_loader_kwargs=file_loader_kwargs,
            search_engines=search_engines,
            simple_se_result_sort=simple_se_result_sort,
            pytesseract_exe_fp=pytesseract_exe_fp,
            td_kwargs=td_kwargs,
            tpe_kwargs=tpe_kwargs,
            ppe_kwargs=ppe_kwargs,
            log_dir=log_dir,
            clean_dir=parsed_file_dir,
            ordinance_file_dir=source_file_dir,
            jurisdiction_dbs_dir=shard_dir,
            perform_se_search=perform_se_search,
            perform_website_search=perform_website_search,
            make_paths_relative=make_paths_relative,
            log_level=log_level,
            keep_async_logs=keep_async_logs,
        )


class ExtractionRequest(BaseRequest):
    """Parameter Object for extraction mode"""

    MODE = COMPASSRunMode.EXTRACT
    """COMPASSRunMode associated with this request type"""

    def __init__(  # ruff:ignore[too-many-arguments]
        self,
        out_dir,
        tech,
        jurisdiction_fp,
        collection_manifest_fp,
        *,
        model="gpt-4o-mini",
        max_num_docs_to_parse_per_jurisdiction=None,
        max_num_concurrent_jurisdictions=25,
        file_loader_kwargs=None,
        td_kwargs=None,
        tpe_kwargs=None,
        ppe_kwargs=None,
        log_dir=None,
        clean_dir=None,
        ordinance_file_dir=None,
        jurisdiction_dbs_dir=None,
        llm_costs=None,
        log_level="INFO",
        keep_async_logs=False,
    ):
        """

        Parameters
        ----------
        out_dir : path-like
            Path to the output directory. If it does not exist, it will
            be created. This directory will contain the saved collection
            manifest, downloaded ordinance documents, parsed document
            text, usage metadata, and default subdirectories for logs
            and intermediate outputs (unless otherwise specified).
        tech : str
            Label indicating which technology type is being processed.
            Must be one of the keys of
            :obj:`~compass.plugin.registry.PLUGIN_REGISTRY`.
        jurisdiction_fp : path-like
            Path to a CSV file specifying the jurisdictions to process.
            The CSV must contain at least two columns: "County" and
            "State", which specify the county and state names,
            respectively. If you would like to process a subdivision
            with a county, you must also include "Subdivision" and
            "Jurisdiction Type" columns. The "Subdivision" should be the
            name of the subdivision, and the "Jurisdiction Type" should
            be a string identifying the type of subdivision (e.g.,
            "City", "Township", etc.)
        collection_manifest_fp : path-like or list of path-like, optional
            Path to the JSON collection manifest created by the document
            collection step. This can be a single path or a list of
            paths for multiple collection manifests, any of which may
            include glob patterns. Each collection manifest must contain
            the persisted document information needed to reload each
            collected document for extraction. By default, ``None``.
        model : str or list of dict, default="gpt-4o-mini"
            LLM model(s) to use for scraping and parsing ordinance
            documents. If a string is provided, it is assumed to be the
            name of the default model (e.g., "gpt-4o"), and environment
            variables are used for authentication.

            If a list is provided, it should contain dictionaries of
            arguments that can initialize instances of
            :class:`~compass.llm.config.OpenAIConfig`. Each dictionary
            can specify the model name, client type, and initialization
            arguments.

            Each dictionary must also include a ``tasks`` key, which
            maps to a string or list of strings indicating the tasks
            that instance should handle. Exactly one of the instances
            **must** include "default" as a task, which will be used
            when no specific task is matched. For example::

                "model": [
                    {
                        "model": "gpt-4o-mini",
                        "llm_call_kwargs": {
                            "temperature": 0,
                            "timeout": 300,
                        },
                        "client_kwargs": {
                            "api_key": "<your_api_key>",
                            "api_version": "<your_api_version>",
                            "azure_endpoint": "<your_azure_endpoint>",
                        },
                        "tasks": ["default", "date_extraction"],
                    },
                    {
                        "model": "gpt-4o",
                        "client_type": "openai",
                        "tasks": ["ordinance_text_extraction"],
                    }
                ]

            .. IMPORTANT::
                You will need to ensure that the model name used here
                matches your deployment if you are using Azure OpenAI.
                For example, if you deployed the GPT-4o-mini model under
                the name ``"gpt-4o-mini-2025-04-11"``, you would want to
                set ``"model": "gpt-4o-mini-2025-04-11"``.

            By default, ``"gpt-4o-mini"``.
        max_num_docs_to_parse_per_jurisdiction : int, optional
            Maximum number of documents to parse for each jurisdiction
            (regardless of the collection method). If ``None``, all
            collected documents are parsed. By default, ``None``.
        max_num_concurrent_jurisdictions : int, default=25
            Maximum number of jurisdictions to process concurrently.
            Limiting this can help manage memory usage when dealing with
            a large number of documents. By default, ``25``.
        file_loader_kwargs : dict, optional
            Dictionary of keyword argument pairs to initialize
            :class:`elm.web.file_loader.AsyncWebFileLoader`. If found,
            the ``"pw_launch_kwargs"`` key in these will also be used to
            initialize the Playwright-backed Google search used for
            search engine retrieval. By default, ``None``.
        td_kwargs : dict, optional
            Additional keyword arguments to pass to
            :class:`tempfile.TemporaryDirectory`. The temporary
            directory is used to store documents which have not yet been
            confirmed to contain relevant information.
            By default, ``None``.
        tpe_kwargs : dict, optional
            Additional keyword arguments to pass to
            :class:`concurrent.futures.ThreadPoolExecutor`, used for
            I/O-bound tasks such as logging and file writes.
            By default, ``None``.
        ppe_kwargs : dict, optional
            Additional keyword arguments to pass to
            :class:`concurrent.futures.ProcessPoolExecutor`, used for
            CPU-bound tasks such as PDF loading and parsing.
            By default, ``None``.
        log_dir : path-like, optional
            Path to the directory for storing log files. If not
            provided, a ``logs`` subdirectory will be created inside
            `out_dir`. By default, ``None``.
        clean_dir : path-like, optional
            Path to the directory for storing cleaned ordinance text
            output. If not provided, a ``cleaned_text`` subdirectory
            will be created inside `out_dir`. By default, ``None``.
        ordinance_file_dir : path-like, optional
            Path to the directory where downloaded ordinance files (PDFs
            or HTML) for each jurisdiction are stored. If not provided,
            a ``ordinance_files`` subdirectory will be created inside
            `out_dir`. By default, ``None``.
        jurisdiction_dbs_dir : path-like, optional
            Path to the directory where parsed ordinance database files
            are stored for each jurisdiction. If not provided, a
            ``jurisdiction_dbs`` subdirectory will be created inside
            `out_dir`. By default, ``None``.
        llm_costs : dict, optional
            Dictionary mapping model names to their token costs, used to
            track the estimated total cost of LLM usage during the run.
            The structure should be::

                {"model_name": {"prompt": float, "response": float}}

            Costs are specified in dollars per million tokens.
            For example::

                "llm_costs": {"my_gpt": {"prompt": 1.5, "response": 3}}

            registers a model named `"my_gpt"` with a cost of $1.5 per
            million input (prompt) tokens and $3 per million output
            (response) tokens for the current processing run.

            .. NOTE::

                The displayed total cost does not track cached tokens,
                so treat it like an estimate. Your final API costs may
                vary.

            If set to ``None``, no custom model costs are recorded, and
            cost tracking may be unavailable in the progress bar.
            By default, ``None``.
        log_level : str, default="INFO"
            Logging level for ordinance scraping and parsing (e.g.,
            "TRACE", "DEBUG", "INFO", "WARNING", or "ERROR").
            By default, ``"INFO"``.
        keep_async_logs : bool, default=False
            Option to store the full asynchronous log record to a file.
            This is only useful if you intend to monitor overall
            processing progress from a file instead of from the
            terminal. If ``True``, all of the unordered records are
            written to a "all.log" file in the `log_dir` directory.
            By default, ``False``.
        """  # ruff:ignore[doc-line-too-long]

        super().__init__(
            out_dir=out_dir,
            tech=tech,
            jurisdiction_fp=jurisdiction_fp,
            model=model,
            max_num_docs_to_parse_per_jurisdiction=(
                max_num_docs_to_parse_per_jurisdiction
            ),
            max_num_concurrent_jurisdictions=max_num_concurrent_jurisdictions,
            file_loader_kwargs=file_loader_kwargs,
            td_kwargs=td_kwargs,
            tpe_kwargs=tpe_kwargs,
            ppe_kwargs=ppe_kwargs,
            log_dir=log_dir,
            clean_dir=clean_dir,
            ordinance_file_dir=ordinance_file_dir,
            jurisdiction_dbs_dir=jurisdiction_dbs_dir,
            log_level=log_level,
            keep_async_logs=keep_async_logs,
            collection_manifest_fp=collection_manifest_fp,
            llm_costs=llm_costs,
        )


class JurisdictionResult:
    """Result Object for one jurisdiction run"""

    def __init__(self, jurisdiction=None, ord_db_fp=None):
        """

        Parameters
        ----------
        jurisdiction : object, optional
            Jurisdiction object associated with this pipeline result.
            By default, ``None``.
        ord_db_fp : path-like, optional
            Path to the ordinance database produced for the
            jurisdiction. By default, ``None``.
        """
        self.jurisdiction = jurisdiction
        self.ord_db_fp = ord_db_fp

    def __bool__(self):
        return self.ord_db_fp is not None


def build_models(user_input, *, allow_empty=False, rate_tracker=None):
    """[NOT PUBLIC API] Build configured model registry"""
    if user_input is None:
        return {} if allow_empty else {LLMTasks.DEFAULT: OpenAIConfig()}

    if isinstance(user_input, str):
        return {LLMTasks.DEFAULT: OpenAIConfig(name=user_input)}

    caller_instances = {}
    for raw_kwargs in user_input:
        for task, model_config in _config_for_tasks(raw_kwargs, rate_tracker):
            _verify_task_not_duplicate(task, caller_instances)
            caller_instances[task] = model_config

    _verify_default_case_handled(caller_instances, allow_empty)
    return caller_instances


def _config_for_tasks(kwargs, rate_tracker=None):
    """Yield (task, model_config) pairs for the given raw kwargs"""
    kwargs = dict(kwargs)
    tasks = kwargs.pop("tasks", LLMTasks.DEFAULT)
    if isinstance(tasks, str):
        tasks = [tasks]

    model_config = OpenAIConfig(**kwargs)
    model_config.llm_call_kwargs.update({"rate_tracker": rate_tracker})
    for task in tasks:
        yield task, model_config


def _verify_task_not_duplicate(task, caller_instances):
    """Verify that the given task has not already been defined"""
    if task in caller_instances:
        msg = (
            f"Found duplicated task: {task!r}. Please ensure "
            "each LLM caller definition has uniquely-assigned "
            "tasks."
        )
        raise COMPASSValueError(msg)


def _verify_default_case_handled(caller_instances, allow_empty):
    """Verify that the default LLM task is handled correctly"""
    if not allow_empty and LLMTasks.DEFAULT not in caller_instances:
        msg = (
            "No 'default' LLM caller defined in the `model` portion "
            "of the input config! Please ensure exactly one of the "
            "model definitions has 'tasks' set to 'default' or left "
            f"unspecified. Found tasks: {list(caller_instances)}"
        )
        raise COMPASSValueError(msg)
