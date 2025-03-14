"""Wind ordinance structured parsing class"""

import asyncio
import logging
from copy import deepcopy
from itertools import chain
from warnings import warn

import pandas as pd

from compass.llm.calling import BaseLLMCaller, ChatLLMCaller
from compass.extraction.features import SetbackFeatures
from compass.extraction.common import (
    EXTRACT_ORIGINAL_TEXT_PROMPT,
    run_async_tree,
    run_async_tree_with_bm,
    found_ord,
    empty_output,
    setup_async_decision_tree,
    setup_base_graph,
    setup_participating_owner,
    setup_graph_extra_restriction,
    setup_graph_permitted_use_districts,
)
from compass.extraction.wind.graphs import (
    setup_graph_wes_types,
    setup_multiplier,
    setup_conditional,
)
from compass.warnings import COMPASSWarning


logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_MESSAGE = (
    "You are a legal scholar informing a wind energy developer about local "
    "zoning ordinances."
)
SETBACKS_SYSTEM_MESSAGE = (
    f"{DEFAULT_SYSTEM_MESSAGE} "
    "For the duration of this conversation, only focus on "
    "ordinances relating to setbacks from {feature} for {tech}. Ignore "
    "all text that pertains to private, micro, small, or medium sized wind "
    "energy systems."
)
RESTRICTIONS_SYSTEM_MESSAGE = (
    f"{DEFAULT_SYSTEM_MESSAGE} "
    "For the duration of this conversation, only focus on "
    "ordinances relating to {restriction} for {tech}. Ignore "
    "all text that pertains to private, micro, small, or medium sized wind "
    "energy systems."
)
PERMITTED_USE_SYSTEM_MESSAGE = (
    f"{DEFAULT_SYSTEM_MESSAGE} "
    "For the duration of this conversation, only focus on permitted uses for "
    "{tech}. Ignore all text that pertains to private, micro, small, or "
    "medium sized wind energy systems."
)
EXTRA_NUMERICAL_RESTRICTIONS = {
    "noise": "maximum noise level allowed",
    "max height": "maximum turbine height allowed",
    "min lot size": "minimum lot, parcel, or tract size allowed",
    "shadow flicker": "maximum shadow flicker allowed",
    "density": "maximum turbine spacing allowed",
    "blade clearance": "minimum blade clearance allowed",
}
EXTRA_QUALITATIVE_RESTRICTIONS = {
    "color": "color or finish requirements",
    "decommissioning": "decommissioning requirements",
    "lighting": "lighting requirements",
    "moratorium": "moratoriums or bans",
    "visual impact": "visual impact assessment requirements",
}


class StructuredWindParser(BaseLLMCaller):
    """Base class for parsing structured data"""

    def _init_chat_llm_caller(self, system_message):
        """Initialize a ChatLLMCaller instance for the DecisionTree"""
        return ChatLLMCaller(
            self.llm_service,
            system_message=system_message,
            usage_tracker=self.usage_tracker,
            **self.kwargs,
        )

    async def _check_wind_turbine_type(self, text):
        """Get the largest turbine size mentioned in the text"""
        logger.debug("Checking turbine_types")
        tree = setup_async_decision_tree(
            setup_graph_wes_types,
            text=text,
            chat_llm_caller=self._init_chat_llm_caller(DEFAULT_SYSTEM_MESSAGE),
        )
        decision_tree_wes_types_out = await run_async_tree(tree)

        return (
            decision_tree_wes_types_out.get("largest_wes_type")
            or "large wind energy systems"
        )


class StructuredWindOrdinanceParser(StructuredWindParser):
    """LLM ordinance document structured data scraping utility

    Purpose:
        Extract structured ordinance data from text.
    Responsibilities:
        1. Extract ordinance values into structured format by executing
           a decision-tree-based chain-of-thought prompt on the text for
           each value to be extracted.
    Key Relationships:
        Uses a :class:`~compass.llm.calling.StructuredLLMCaller` for
        LLM queries and multiple
        :class:`~compass.extraction.tree.AsyncDecisionTree` instances
        to guide the extraction of individual values.
    """

    async def parse(self, text):
        """Parse text and extract structure ordinance data

        Parameters
        ----------
        text : str
            Ordinance text which may or may not contain setbacks for one
            or more features (property lines, structure, roads, etc.).
            Text can also contain other supported regulations (noise,
            shadow-flicker, etc,) which will be extracted as well.

        Returns
        -------
        pd.DataFrame
            DataFrame containing parsed-out ordinance values.
        """
        largest_wes_type = await self._check_wind_turbine_type(text)
        logger.info("Largest WES type found in text: %s", largest_wes_type)

        outer_task_name = asyncio.current_task().get_name()
        feature_parsers = [
            asyncio.create_task(
                self._parse_setback_feature(
                    text, feature_kwargs, largest_wes_type
                ),
                name=outer_task_name,
            )
            for feature_kwargs in SetbackFeatures()
        ]
        extras_parsers = [
            asyncio.create_task(
                self._parse_extra_restriction(
                    text, feature, r_text, largest_wes_type, is_numerical=True
                ),
                name=outer_task_name,
            )
            for feature, r_text in EXTRA_NUMERICAL_RESTRICTIONS.items()
        ]
        extras_parsers += [
            asyncio.create_task(
                self._parse_extra_restriction(
                    text, feature, r_text, largest_wes_type, is_numerical=False
                ),
                name=outer_task_name,
            )
            for feature, r_text in EXTRA_QUALITATIVE_RESTRICTIONS.items()
        ]
        outputs = await asyncio.gather(*(feature_parsers + extras_parsers))

        return pd.DataFrame(chain.from_iterable(outputs))

    async def _parse_extra_restriction(
        self, text, feature, restriction_text, largest_wes_type, is_numerical
    ):
        """Parse a non-setback restriction from the text"""
        logger.debug("Parsing extra feature %r", feature)
        system_message = RESTRICTIONS_SYSTEM_MESSAGE.format(
            restriction=restriction_text, tech=largest_wes_type
        )
        tree = setup_async_decision_tree(
            setup_graph_extra_restriction,
            is_numerical=is_numerical,
            tech=largest_wes_type,
            restriction=restriction_text,
            text=text,
            chat_llm_caller=self._init_chat_llm_caller(system_message),
        )
        info = await run_async_tree(tree)
        info.update({"feature": feature, "quantitative": is_numerical})
        return [info]

    async def _parse_setback_feature(
        self, text, feature_kwargs, largest_wes_type
    ):
        """Parse values for a setback feature"""
        feature = feature_kwargs["feature_id"]
        feature_kwargs["tech"] = largest_wes_type
        logger.debug("Parsing feature %r", feature)

        base_messages = await self._base_messages(text, **feature_kwargs)
        if not found_ord(base_messages):
            logger.debug("Failed `found_ord` check for feature %r", feature)
            return empty_output(feature)

        if feature not in {"struct", "prop_line"}:
            output = {"feature": feature}
            output.update(
                await self._extract_setback_values(
                    text,
                    base_messages=base_messages,
                    **feature_kwargs,
                )
            )
            return [output]

        return await self._extract_setback_values_for_p_or_np(
            text, base_messages, **feature_kwargs
        )

    async def _base_messages(self, text, **feature_kwargs):
        """Get base messages for setback feature parsing"""
        system_message = SETBACKS_SYSTEM_MESSAGE.format(
            feature=feature_kwargs["feature"],
            tech=feature_kwargs["tech"],
        )
        tree = setup_async_decision_tree(
            setup_base_graph,
            text=text,
            chat_llm_caller=self._init_chat_llm_caller(system_message),
            **feature_kwargs,
        )
        await run_async_tree(tree, response_as_json=False)
        return deepcopy(tree.chat_llm_caller.messages)

    async def _extract_setback_values_for_p_or_np(
        self, text, base_messages, **feature_kwargs
    ):
        """Extract setback values for participating ordinances"""
        logger.debug("Checking participating vs non-participating")
        decision_tree_participating_out = await self._run_setback_graph(
            setup_participating_owner,
            text,
            base_messages=deepcopy(base_messages),
            **feature_kwargs,
        )
        outer_task_name = asyncio.current_task().get_name()
        p_or_np_parsers = [
            asyncio.create_task(
                self._parse_p_or_np_text(
                    key, sub_text, base_messages, **feature_kwargs
                ),
                name=outer_task_name,
            )
            for key, sub_text in decision_tree_participating_out.items()
        ]
        return await asyncio.gather(*p_or_np_parsers)

    async def _parse_p_or_np_text(
        self, key, sub_text, base_messages, **feature_kwargs
    ):
        """Parse participating sub-text for ord values"""
        feature = feature_kwargs["feature_id"]
        out_feat_name = f"{feature} ({key})"
        output = {"feature": out_feat_name}
        if not sub_text:
            return output

        feature = feature_kwargs["feature"]
        feature = f"{key} {feature}"
        feature_kwargs["feature"] = feature

        base_messages = deepcopy(base_messages)
        base_messages[-2]["content"] = EXTRACT_ORIGINAL_TEXT_PROMPT.format(
            feature=feature, tech=feature_kwargs["tech"]
        )
        base_messages[-1]["content"] = sub_text

        values = await self._extract_setback_values(
            sub_text,
            base_messages=base_messages,
            **feature_kwargs,
        )
        output.update(values)
        return output

    async def _extract_setback_values(self, text, **kwargs):
        """Extract setback values for a given feature from input text"""
        decision_tree_out = await self._run_setback_graph(
            setup_multiplier, text, **kwargs
        )
        decision_tree_out = _update_output_keys(decision_tree_out)

        if decision_tree_out.get("value") is None:
            return decision_tree_out

        decision_tree_conditional_out = await self._run_setback_graph(
            setup_conditional, text, **kwargs
        )
        decision_tree_out.update(decision_tree_conditional_out)
        return decision_tree_out

    async def _run_setback_graph(
        self,
        graphs_setup_func,
        text,
        feature,
        tech,
        base_messages=None,
        **kwargs,
    ):
        """Generic function to run async tree"""
        system_message = SETBACKS_SYSTEM_MESSAGE.format(
            feature=feature, tech=tech
        )
        tree = setup_async_decision_tree(
            graphs_setup_func,
            feature=feature,
            text=text,
            chat_llm_caller=self._init_chat_llm_caller(system_message),
            **kwargs,
        )
        if base_messages:
            return await run_async_tree_with_bm(tree, base_messages)
        return await run_async_tree(tree)


class StructuredWindPermittedUseDistrictsParser(StructuredWindParser):
    """LLM permitted use districts  scraping utility

    Purpose:
        Extract structured ordinance data from text.
    Responsibilities:
        1. Extract ordinance values into structured format by executing
           a decision-tree-based chain-of-thought prompt on the text for
           each value to be extracted.
    Key Relationships:
        Uses a :class:`~compass.llm.calling.StructuredLLMCaller` for
        LLM queries and multiple
        :class:`~compass.extraction.tree.AsyncDecisionTree` instances
        to guide the extraction of individual values.
    """

    _LARGE_WES_CLARIFICATION = (
        "Large wind energy systems (WES) may also be referred to as wind "
        "turbines, wind energy conversion systems (WECS), wind energy "
        "facilities (WEF), wind energy turbines (WET), large wind energy "
        "turbines (LWET), utility-scale wind energy turbines (UWET), "
        "commercial wind energy conversion systems (CWECS), alternate "
        "energy systems (AES), or similar. "
    )
    _USE_TYPES = [
        {
            "feature_id": "primary use districts",
            "use_type": "primary use",
        },
        {
            "feature_id": "special use districts",
            "use_type": "special use",
        },
        {
            "feature_id": "accessory use districts",
            "use_type": "accessory use",
        },
    ]

    async def parse(self, text):
        """Parse text and extract permitted use districts data

        Parameters
        ----------
        text : str
            Permitted use districts text which may or may not contain
            information about allowed uses in one or more districts.

        Returns
        -------
        pd.DataFrame
            DataFrame containing parsed-out allowed-use district names.
        """
        largest_wes_type = await self._check_wind_turbine_type(text)
        logger.info("Largest WES type found in text: %s", largest_wes_type)

        outer_task_name = asyncio.current_task().get_name()
        feature_parsers = [
            asyncio.create_task(
                self._parse_permitted_use_districts(
                    text, largest_wes_type, **use_type_kwargs
                ),
                name=outer_task_name,
            )
            for use_type_kwargs in self._USE_TYPES
        ]
        outputs = await asyncio.gather(*(feature_parsers))

        return pd.DataFrame(chain.from_iterable(outputs))

    async def _parse_permitted_use_districts(
        self, text, largest_wes_type, feature_id, use_type
    ):
        """Parse a non-setback restriction from the text"""
        logger.debug("Parsing use type: %r", feature_id)
        system_message = PERMITTED_USE_SYSTEM_MESSAGE.format(
            tech=largest_wes_type
        )
        tree = setup_async_decision_tree(
            setup_graph_permitted_use_districts,
            tech=largest_wes_type,
            clarifications=self._LARGE_WES_CLARIFICATION,
            text=text,
            use_type=use_type,
            chat_llm_caller=self._init_chat_llm_caller(system_message),
        )
        info = await run_async_tree(tree)
        info.update({"feature": feature_id, "quantitative": True})
        return [info]


def _update_output_keys(output):
    """Standardize output keys

    We could standardize output keys by modifying the LLM prompts, but
    have found that it's more accurate to instruct the LLM to use
    descriptive keys (e.g. "mult_value" instead of "value" or
    "mult_type" instead of "units")
    """

    if "mult_value" not in output:
        return output

    output["value"] = output.pop("mult_value")

    if units := output.get("units"):
        msg = f"Found non-null units value for multiplier: {units}"
        warn(msg, COMPASSWarning)
    output["units"] = output.pop("mult_type", None)

    return output
