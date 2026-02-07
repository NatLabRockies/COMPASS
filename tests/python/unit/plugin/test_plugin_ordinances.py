"""COMPASS ordinance plugin tests"""

from pathlib import Path

import pytest

from compass.plugin.ordinance import (
    BaseTextCollector,
    BaseTextExtractor,
    BaseParser,
    OrdinanceExtractionPlugin,
)
from compass.exceptions import COMPASSPluginConfigurationError


def test_plugin_validation_parse_key_same():
    """Test plugin interface validation logic"""

    class COLL1(BaseTextCollector):
        OUT_LABEL = "collected"

    class EXT1(BaseTextExtractor):
        IN_LABEL = "collected"
        OUT_LABEL = "extracted"

    class EXT2(BaseTextExtractor):
        IN_LABEL = "collected"
        OUT_LABEL = "extracted_2"

    class PARS1(BaseParser):
        IN_LABEL = "extracted"
        OUT_LABEL = "parsed_1"

    class PARS2(BaseParser):
        IN_LABEL = "collected"
        OUT_LABEL = "parsed_1"

    class MYPlugin(OrdinanceExtractionPlugin):
        TEXT_COLLECTORS = [COLL1]
        TEXT_EXTRACTORS = [EXT1, EXT2]
        PARSERS = [PARS1, PARS2]

        IDENTIFIER = "test"
        WEBSITE_KEYWORDS = ["test"]
        QUESTION_TEMPLATES = ["test"]
        heuristic = None

        async def parse_docs_for_structured_data(self, extraction_context):
            return extraction_context

    with pytest.raises(
        COMPASSPluginConfigurationError,
        match="Multiple processing classes produce the same OUT_LABEL key",
    ):
        MYPlugin(None, None, None).validate_plugin_configuration()


def test_plugin_validation_extract_key_same():
    """Test plugin interface validation logic"""

    class COLL1(BaseTextCollector):
        OUT_LABEL = "collected"

    class EXT1(BaseTextExtractor):
        IN_LABEL = "collected"
        OUT_LABEL = "extracted"

    class EXT2(BaseTextExtractor):
        IN_LABEL = "collected"
        OUT_LABEL = "extracted"

    class PARS1(BaseParser):
        IN_LABEL = "extracted"
        OUT_LABEL = "parsed_1"

    class PARS2(BaseParser):
        IN_LABEL = "collected"
        OUT_LABEL = "parsed_2"

    class MYPlugin(OrdinanceExtractionPlugin):
        TEXT_COLLECTORS = [COLL1]
        TEXT_EXTRACTORS = [EXT1, EXT2]
        PARSERS = [PARS1, PARS2]

        IDENTIFIER = "test"
        WEBSITE_KEYWORDS = ["test"]
        QUESTION_TEMPLATES = ["test"]
        heuristic = None

        async def parse_docs_for_structured_data(self, extraction_context):
            return extraction_context

    with pytest.raises(
        COMPASSPluginConfigurationError,
        match="Multiple processing classes produce the same OUT_LABEL key",
    ):
        MYPlugin(None, None, None).validate_plugin_configuration()


def test_plugin_validation_no_in_key_for_extract():
    """Test plugin interface validation logic"""

    class COLL1(BaseTextCollector):
        OUT_LABEL = "collected"

    class EXT1(BaseTextExtractor):
        IN_LABEL = "collected"
        OUT_LABEL = "extracted"

    class EXT2(BaseTextExtractor):
        IN_LABEL = "collected_2"
        OUT_LABEL = "extracted_1"

    class PARS1(BaseParser):
        IN_LABEL = "extracted"
        OUT_LABEL = "parsed_1"

    class PARS2(BaseParser):
        IN_LABEL = "collected"
        OUT_LABEL = "parsed_2"

    class MYPlugin(OrdinanceExtractionPlugin):
        TEXT_COLLECTORS = [COLL1]
        TEXT_EXTRACTORS = [EXT1, EXT2]
        PARSERS = [PARS1, PARS2]

        IDENTIFIER = "test"
        WEBSITE_KEYWORDS = ["test"]
        QUESTION_TEMPLATES = ["test"]
        heuristic = None

        async def parse_docs_for_structured_data(self, extraction_context):
            return extraction_context

    with pytest.raises(
        COMPASSPluginConfigurationError,
        match=(
            r"One or more processing classes require IN_LABEL 'collected_2', "
            r"which is not produced by any previous processing class: "
            r"\['EXT2'\]"
        ),
    ):
        MYPlugin(None, None, None).validate_plugin_configuration()


def test_plugin_validation_no_in_key_for_parse():
    """Test plugin interface validation logic"""

    class COLL1(BaseTextCollector):
        OUT_LABEL = "collected"

    class EXT1(BaseTextExtractor):
        IN_LABEL = "collected"
        OUT_LABEL = "extracted"

    class EXT2(BaseTextExtractor):
        IN_LABEL = "collected"
        OUT_LABEL = "extracted_1"

    class PARS1(BaseParser):
        IN_LABEL = "extracted"
        OUT_LABEL = "parsed_1"

    class PARS2(BaseParser):
        IN_LABEL = "collected_2"
        OUT_LABEL = "parsed_2"

    class MYPlugin(OrdinanceExtractionPlugin):
        TEXT_COLLECTORS = [COLL1]
        TEXT_EXTRACTORS = [EXT1, EXT2]
        PARSERS = [PARS1, PARS2]

        IDENTIFIER = "test"
        WEBSITE_KEYWORDS = ["test"]
        QUESTION_TEMPLATES = ["test"]
        heuristic = None

        async def parse_docs_for_structured_data(self, extraction_context):
            return extraction_context

    with pytest.raises(
        COMPASSPluginConfigurationError,
        match=(
            r"One or more processing classes require IN_LABEL 'collected_2', "
            r"which is not produced by any previous processing class: "
            r"\['PARS2'\]"
        ),
    ):
        MYPlugin(None, None, None).validate_plugin_configuration()


if __name__ == "__main__":
    pytest.main(["-q", "--show-capture=all", Path(__file__), "-rapP"])
