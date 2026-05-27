"""Data classes used for the COMPASS pipeline"""

from copy import deepcopy
from functools import cached_property

from elm.web.search.run import SEARCH_ENGINE_OPTIONS

from compass.utilities.enums import COMPASSRunMode


class SearchSettings:
    """Value Object for search and crawl settings"""

    def __init__(
        self,
        num_urls_to_check_per_jurisdiction=5,
        max_num_concurrent_browsers=10,
        max_num_concurrent_website_searches=10,
        url_ignore_substrings=None,
        search_engines=None,
        pytesseract_exe_fp=None,
    ):
        """

        Parameters
        ----------
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
        url_ignore_substrings : list of str, optional
            A list of substrings that, if found in any URL, will cause
            the URL to be excluded from consideration. This can be used
            to specify particular websites or entire domains to ignore.
            By default, ``None``.
        search_engines : list, optional
            A list of dictionaries describing the search engine classes
            and keyword arguments to use for search engine retrieval. If
            ``None``, the default search engine configurations and
            fallback order are used. By default, ``None``.
        pytesseract_exe_fp : path-like, optional
            Path to the `pytesseract` executable. If specified, OCR will
            be used to extract text from scanned PDFs using Google's
            Tesseract. By default, ``None``.
        """
        self.num_urls_to_check_per_jurisdiction = (
            num_urls_to_check_per_jurisdiction
        )
        self.max_num_concurrent_browsers = max_num_concurrent_browsers
        self.max_num_concurrent_website_searches = (
            max_num_concurrent_website_searches
        )
        self.url_ignore_substrings = url_ignore_substrings
        self.search_engines = search_engines
        self.pytesseract_exe_fp = pytesseract_exe_fp


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


class ModelSelection:
    """Value Object for model selection and cost settings"""

    def __init__(self, model="gpt-4o-mini", llm_costs=None):
        """

        Parameters
        ----------
        model : str or list of dict, default="gpt-4o-mini"
            Optional model configuration used only for collection-side
            LLM tasks, such as validating a user-supplied jurisdiction
            website. If provided as a string, it is treated as the
            default model name. If provided as a list, each entry
            should contain keyword arguments used to initialize
            :class:`~compass.llm.config.OpenAIConfig`, along with a
            ``tasks`` key describing which LLM tasks that configuration
            should handle. By default, ``None``.
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
        """
        self.model = model
        self.llm_costs = llm_costs


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
        url_ignore_substrings=None,
        pytesseract_exe_fp=None,
        search_engines=None,
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
        url_ignore_substrings : list of str, optional
            A list of substrings that, if found in any URL, will cause
            the URL to be excluded from consideration. This can be used
            to specify particular websites or entire domains to ignore.
            For example::

                url_ignore_substrings = [
                    "wikipedia",
                    "nlr.gov",
                    "www.co.delaware.in.us/documents/1649699794_0382.pdf",
                ]

            The above configuration would ignore all `wikipedia`
            articles, all websites on the NLR domain, and the specific
            file located at
            `www.co.delaware.in.us/documents/1649699794_0382.pdf`.
            By default, ``None``.
        pytesseract_exe_fp : path-like, optional
            Path to the `pytesseract` executable. If specified, OCR will
            be used to extract text from scanned PDFs using Google's
            Tesseract. By default ``None``.
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
        """
        self.num_urls_to_check_per_jurisdiction = (
            num_urls_to_check_per_jurisdiction
        )
        self.max_num_concurrent_browsers = max_num_concurrent_browsers
        self.max_num_concurrent_website_searches = (
            max_num_concurrent_website_searches
        )
        self.url_ignore_substrings = url_ignore_substrings
        self.pytesseract_exe_fp = pytesseract_exe_fp
        self._search_engines_input = search_engines

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


class BaseRequest:
    """Parameter Object base class for pipeline requests"""

    MODE = None
    """COMPASSRunMode associated with this request type"""

    def __init__(  # noqa: PLR0913
        self,
        out_dir,
        tech,
        jurisdiction_fp,
        *,
        model="gpt-4o-mini",
        llm_costs=None,
        num_urls_to_check_per_jurisdiction=5,
        max_num_concurrent_browsers=10,
        max_num_concurrent_website_searches=10,
        max_num_concurrent_jurisdictions=25,
        url_ignore_substrings=None,
        known_local_docs=None,
        known_doc_urls=None,
        file_loader_kwargs=None,
        search_engines=None,
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
            Optional model configuration used only for collection-side
            LLM tasks, such as validating a user-supplied jurisdiction
            website. If provided as a string, it is treated as the
            default model name. If provided as a list, each entry
            should contain keyword arguments used to initialize
            :class:`~compass.llm.config.OpenAIConfig`, along with a
            ``tasks`` key describing which LLM tasks that configuration
            should handle. By default, ``None``.
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
        url_ignore_substrings : list of str, optional
            A list of substrings that, if found in any URL, will cause
            the URL to be excluded from consideration. This can be used
            to specify particular websites or entire domains to ignore.
            By default, ``None``.
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
            By default, ``True``.
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
        collection_manifest_fp : path-like, optional
            Path to the JSON collection manifest created by the document
            collection step. The manifest must contain the persisted
            document information needed to reload each collected
            document for extraction. Only needed if running in
            extraction mode with a separate collection step.
            By default, ``None``.
        """
        self.tech = tech
        self.jurisdiction_fp = jurisdiction_fp
        self.perform_se_search = perform_se_search
        self.perform_website_search = perform_website_search
        self.collection_manifest_fp = collection_manifest_fp
        self.file_loader_kwargs = file_loader_kwargs

        self.search_settings = SearchSettings(
            num_urls_to_check_per_jurisdiction=(
                num_urls_to_check_per_jurisdiction
            ),
            max_num_concurrent_browsers=max_num_concurrent_browsers,
            max_num_concurrent_website_searches=(
                max_num_concurrent_website_searches
            ),
            url_ignore_substrings=url_ignore_substrings,
            search_engines=search_engines,
            pytesseract_exe_fp=pytesseract_exe_fp,
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
        self.model_selection = ModelSelection(model=model, llm_costs=llm_costs)


class ProcessRequest(BaseRequest):
    """Parameter Object for full process mode"""

    MODE = COMPASSRunMode.PROCESS
    """COMPASSRunMode associated with this request type"""


class CollectionRequest(BaseRequest):
    """Parameter Object for collection mode"""

    MODE = COMPASSRunMode.COLLECT
    """COMPASSRunMode associated with this request type"""

    def __init__(self, out_dir, tech, jurisdiction_fp, **kwargs):
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
        **kwargs : dict
            Additional keyword arguments forwarded to
            :class:`BaseRequest`. If not supplied, collection mode sets
            `model` to ``None`` and `make_paths_relative` to ``True``.
        """
        kwargs.setdefault("model", None)
        kwargs.setdefault("make_paths_relative", True)
        super().__init__(out_dir, tech, jurisdiction_fp, **kwargs)


class ExtractionRequest(BaseRequest):
    """Parameter Object for extraction mode"""

    MODE = COMPASSRunMode.EXTRACT
    """COMPASSRunMode associated with this request type"""

    def __init__(
        self, out_dir, tech, jurisdiction_fp, collection_manifest_fp, **kwargs
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
        collection_manifest_fp : path-like
            Path to the JSON collection manifest created by the document
            collection step. The manifest must contain the persisted
            document information needed to reload each collected
            document for extraction.
        **kwargs : dict
            Additional keyword arguments forwarded to
            :class:`BaseRequest`.
        """
        super().__init__(
            out_dir,
            tech,
            jurisdiction_fp,
            collection_manifest_fp=collection_manifest_fp,
            **kwargs,
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
