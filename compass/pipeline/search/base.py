"""Search workflow for jurisdictions"""

from compass.pb import COMPASS_PB
from compass.web.search import search_single_jurisdiction
from compass.pipeline.search.persistence import write_search_manifest_shard


class SearchEngineLinkCollection:
    """Workflow object that uses a search engine to search for docs"""

    def __init__(self, workflow):
        self.workflow = workflow

    async def execute(self, query_templates):
        """Search for URLs and persist one jurisdiction's results

        Parameters
        ----------
        query_templates : iterable of str
            Query templates to format for this jurisdiction.

        Returns
        -------
        dict
            Ranked search results and jurisdiction metadata.
        """
        runtime = self.workflow.runtime
        jurisdiction = self.workflow.jurisdiction
        search_params = runtime.search_params
        COMPASS_PB.update_jurisdiction_task(
            jurisdiction.full_name,
            description="Searching for document URLs...",
        )
        results = await search_single_jurisdiction(
            query_templates,
            jurisdiction,
            search_params.num_urls_to_check_per_jurisdiction,
            runtime.search_engine_semaphore,
            search_params.url_ignore_substrings,
            search_params.url_keep_substrings,
            simple=search_params.simple_se_result_sort,
            **search_params.se_kwargs,
        )
        results["tech"] = runtime.tech
        await write_search_manifest_shard(
            runtime.dirs.se_shards, results, jurisdiction
        )
        return results
