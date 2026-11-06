.. _glossary:

Glossary
========

.. glossary::
   :sorted:

   INFRA-COMPASS
      End-to-end pipeline that discovers, parses, and validates energy
      infrastructure ordinances with LLM tooling.

   LLM
      Large Language Model that interprets ordinance text, classifies
      features, and answers structured extraction questions.

   OCR
      Optical Character Recognition stage powered by ``pytesseract``
      that converts scanned ordinance PDFs into searchable text.

   Pixi
      Environment manager used to install dependencies, run tasks, and
      maintain reproducible shells for COMPASS.

   Playwright
      Browser automation framework used to crawl web portals and
      download ordinance documents reliably.

   analysis run
      Complete invocation of ``compass process`` that ingests a
      configuration file, processes jurisdictions, and writes results to
      the run directory.

   clean directory
      Intermediate folder storing cleaned ordinance text used for LLM
      prompting during feature extraction.

   clean text file
      Plain-text excerpt derived from ordinance documents that isolates
      relevant sections for prompts and validation.

   compass process
      CLI command that executes the end-to-end pipeline using the inputs
      defined in the configuration file.

   configuration file
      JSON or JSON5 document that declares inputs, model assignments,
      concurrency, and output directories for a run.

   decision tree prompt
      Structured prompt template that guides the LLM through branching
      questions to extract quantitative and qualitative ordinance data.

   decision tree
      Hierarchical rubric of questions and outcomes that organizes how
      ordinance features are extracted and validated.

   extraction pipeline
      Crawlers, parsers, and feature detectors that transform raw
      ordinance text into structured records.

   jurisdiction
      County or municipality defined in the jurisdiction CSV that
      frames the geographic scope of an analysis run.

   jurisdiction CSV
      Input spreadsheet whose ``County`` and ``State`` columns list the
      locations processed in a run.

   location
      Combination of county and state identifiers that maps to one row
      in the jurisdiction CSV and produces a single output bundle.

   location file log
      Per-location structured log that aggregates runtime diagnostics
      and JSON exception summaries.

   location manifest
      JSON metadata file emitted per location summarizing source
      documents, extraction status, and validation outcomes.

   log directory
      Folder defined by ``log_dir`` that stores run-level logs, prompt
      archives, and timing summaries.

   llm cost tracker
      Runtime utility that multiplies token usage by configured pricing
      to report estimated spend per model.

   llm service
      Abstraction over providers such as OpenAI or Azure OpenAI that
      enforces authentication, rate limits, and retry policies.

   llm service rate limit
      Configuration value that caps tokens per minute for a model to
      avoid provider throttling.

   llm task
      Logical label assigned to prompt templates that maps to a specific
      model entry within the configuration.

   ordinance
      Legal text that governs energy infrastructure within a
      jurisdiction and feeds the extraction workflows.

   ordinance document
      Source PDF or HTML retrieved during crawling that contains the
      legal language for the targeted technology.

   ordinance file directory
      Folder defined by ``ordinance_file_dir`` that caches downloaded
      ordinance PDFs and HTML files.

   out directory
      Root folder defined by ``out_dir`` where structured results,
      cleaned text, and logs for each run are written.

   ``pytesseract``
      Python wrapper for the Tesseract OCR engine used to enable text
      extraction from scanned ordinance documents.

   rate limiter
      Token-based throttle that keeps LLM requests within provider
      quotas while maximizing throughput.

   structured record
      Tabular representation of ordinance features, thresholds, and
      metadata exported for downstream analysis.

   technology
      ``tech`` configuration key that defines the target infrastructure
      domain, such as solar or wind.

   text splitter
      Utility that chunks ordinance text into overlapping segments sized
      for LLM context windows.

   validation pipeline
      Post-processing stage that verifies extracted features, resolves
      conflicts, and confirms location metadata.

   web search
      Search-and-crawl phase that discovers ordinance links using
      providers such as Tavily, DuckDuckGo Search, or custom engines.
