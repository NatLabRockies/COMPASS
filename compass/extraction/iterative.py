"""LangGraph-based iterative extraction refinement workflow"""

import time
import logging
from typing import TypedDict, Annotated
from operator import add

from langgraph.graph import StateGraph, END

from compass.extraction.iterative_validation import ExtractionValidator
from compass.extraction.iterative_text_filter import FocusedTextFilter
from compass.extraction.iterative_metadata import create_iteration_metadata


logger = logging.getLogger(__name__)


class IterativeExtractionState(TypedDict):
    """State for iterative extraction workflow"""

    iteration: int
    max_iterations: int
    original_text: str
    extraction_schema: dict
    current_extraction: dict
    issues: list[dict]
    metadata_history: Annotated[list[dict], add]
    config: dict
    is_valid: bool
    parser: object
    llm_service: object
    usage_tracker: object
    iteration_start_time: float


class IterativeExtractionGraph:
    """[NOT PUBLIC API]"""

    def __init__(self, config, llm_service, usage_tracker=None):
        """
        Initialize iterative extraction graph

        Parameters
        ----------
        config : dict
            Configuration including max_iterations, validation_strictness,
            text_filter_strategy, and merge_strategy
        llm_service : compass.services.base.Service
            LLM service for validation and re-extraction calls
        usage_tracker : compass.services.usage.UsageTracker, optional
            Token usage tracker. By default, ``None``
        """
        self.config = config
        self.llm_service = llm_service
        self.usage_tracker = usage_tracker
        self._workflow = self._build_workflow()

    def _build_workflow(self):
        """[NOT PUBLIC API]"""
        graph = StateGraph(IterativeExtractionState)

        graph.add_node("validate", self._validate_node)
        graph.add_node("identify", self._identify_node)
        graph.add_node("filter", self._filter_node)
        graph.add_node("reextract", self._reextract_node)
        graph.add_node("merge", self._merge_node)
        graph.add_node("finalize", self._finalize_node)

        graph.set_entry_point("validate")

        graph.add_conditional_edges(
            "validate",
            self._should_continue,
            {
                "continue": "identify",
                "finalize": "finalize",
            },
        )

        graph.add_edge("identify", "filter")
        graph.add_edge("filter", "reextract")
        graph.add_edge("reextract", "merge")
        graph.add_edge("merge", "validate")
        graph.add_edge("finalize", END)

        return graph.compile()

    async def run(self, text, schema, initial_extraction, parser):
        """
        Run iterative refinement workflow

        Parameters
        ----------
        text : str
            Original ordinance text
        schema : dict
            Extraction schema (feature descriptions for dtree,
            full JSON schema for one-shot)
        initial_extraction : dict or pandas.DataFrame
            Results from initial extraction pass
        parser : BaseParser
            Parser instance capable of focused re-extraction

        Returns
        -------
        dict
            Contains "extraction" (refined output) and "metadata"
            (iteration details)
        """
        initial_state = {
            "iteration": 0,
            "max_iterations": self.config.get("max_iterations", 5),
            "original_text": text,
            "extraction_schema": schema,
            "current_extraction": initial_extraction,
            "issues": [],
            "metadata_history": [],
            "config": self.config,
            "is_valid": False,
            "parser": parser,
            "llm_service": self.llm_service,
            "usage_tracker": self.usage_tracker,
            "iteration_start_time": time.time(),
        }

        logger.info("Starting iterative extraction refinement workflow")

        try:
            final_state = await self._workflow.ainvoke(initial_state)
        except Exception as e:
            logger.exception("Error during iterative extraction workflow")
            return {
                "extraction": initial_extraction,
                "metadata": {
                    "iterations": 0,
                    "history": [],
                    "error": str(e),
                },
            }

        logger.info(
            "Iterative extraction complete after %d iterations",
            final_state["iteration"],
        )

        return {
            "extraction": final_state["current_extraction"],
            "metadata": {
                "iterations": final_state["iteration"],
                "history": final_state["metadata_history"],
                "issues_resolved": sum(
                    len(m.get("features_corrected", []))
                    for m in final_state["metadata_history"]
                ),
                "final_valid": final_state["is_valid"],
            },
        }

    async def _validate_node(self, state):
        """[NOT PUBLIC API]"""
        logger.info("Validating extraction (iteration %d)", state["iteration"])

        validator = ExtractionValidator(
            llm_service=state["llm_service"],
            usage_tracker=state["usage_tracker"],
        )

        strictness = state["config"].get("validation_strictness", "moderate")

        validation_result = await validator.validate(
            text=state["original_text"],
            schema=state["extraction_schema"],
            extraction=state["current_extraction"],
            strictness=strictness,
        )

        state["is_valid"] = validation_result["is_valid"]
        state["issues"] = validation_result["issues"]

        logger.info(
            "Validation result: %s (%d issues)",
            "VALID" if state["is_valid"] else "INVALID",
            len(state["issues"]),
        )

        return state

    async def _identify_node(self, state):
        """[NOT PUBLIC API]"""
        logger.info("Identifying problematic features")

        features_to_reextract = list(
            {issue["feature"] for issue in state["issues"]}
        )

        logger.info("Features to re-extract: %s", features_to_reextract)

        state["features_to_reextract"] = features_to_reextract

        return state

    async def _filter_node(self, state):
        """[NOT PUBLIC API]"""
        logger.info("Filtering text for focused re-extraction")

        strategy = state["config"].get("text_filter_strategy", "hybrid")
        context_window = state["config"].get("context_window", 2)

        text_filter = FocusedTextFilter(
            strategy=strategy,
            context_window=context_window,
            llm_service=state["llm_service"],
            usage_tracker=state["usage_tracker"],
        )

        filtered_texts = await text_filter.filter_for_features(
            text=state["original_text"],
            feature_list=state["features_to_reextract"],
            schema=state["extraction_schema"],
        )

        state["filtered_texts"] = filtered_texts

        return state

    async def _reextract_node(self, state):
        """[NOT PUBLIC API]"""
        logger.info("Re-extracting focused features")

        parser = state["parser"]

        if not hasattr(parser, "reextract_focused_features"):
            logger.warning(
                "Parser %s does not support focused re-extraction, skipping",
                parser.__class__.__name__,
            )
            state["reextracted_values"] = {}
            return state

        try:
            reextracted = await parser.reextract_focused_features(
                text_by_feature=state["filtered_texts"],
                feature_list=state["features_to_reextract"],
                schema_info=state["extraction_schema"],
                original_extraction=state["current_extraction"],
            )

            state["reextracted_values"] = reextracted

            logger.info("Re-extracted %d features", len(reextracted))

        except NotImplementedError:
            logger.warning(
                "Parser %s reextract_focused_features not implemented",
                parser.__class__.__name__,
            )
            state["reextracted_values"] = {}
        except Exception:
            logger.exception("Error during focused re-extraction")
            state["reextracted_values"] = {}

        return state

    async def _merge_node(self, state):
        """[NOT PUBLIC API]"""
        logger.info("Merging re-extracted values with original extraction")

        merge_strategy = state["config"].get("merge_strategy", "replace")

        current = state["current_extraction"]
        reextracted = state.get("reextracted_values", {})

        if merge_strategy == "replace":
            if isinstance(current, dict):
                current.update(reextracted)
            else:
                logger.warning(
                    "Merge not implemented for non-dict extractions"
                )

        features_corrected = list(reextracted.keys())

        elapsed = time.time() - state["iteration_start_time"]

        iteration_meta = create_iteration_metadata(
            iteration_num=state["iteration"],
            features_reviewed=list(state["extraction_schema"].keys()),
            issues_found=state["issues"],
            features_corrected=features_corrected,
            llm_calls=0,
            tokens_used={},
            time_elapsed=elapsed,
        )

        state["metadata_history"].append(iteration_meta.__dict__)

        state["iteration"] += 1
        state["iteration_start_time"] = time.time()

        return state

    async def _finalize_node(self, state):
        """[NOT PUBLIC API]"""
        logger.info("Finalizing iterative extraction")
        return state

    def _should_continue(self, state):
        """[NOT PUBLIC API]"""
        if state["is_valid"]:
            logger.info("Extraction is valid, finalizing")
            return "finalize"

        if state["iteration"] >= state["max_iterations"]:
            logger.info("Reached max iterations, finalizing")
            return "finalize"

        if not state["issues"]:
            logger.info("No issues found, finalizing")
            return "finalize"

        logger.info("Continuing to next iteration")
        return "continue"
