"""COMPASS plugin tools"""

from .base import BaseExtractionPlugin
from .interface import (
    BaseHeuristic,
    BaseTextCollector,
    FilteredExtractionPlugin,
)
from .ordinance import (
    BaseTextExtractor,
    BaseParser,
    DocSelectionMethod,
    KeywordBasedHeuristic,
    PromptBasedTextCollector,
    PromptBasedTextExtractor,
    OrdinanceParser,
    OrdinanceExtractionPlugin,
)
from .noop import NoOpHeuristic, NoOpTextCollector, NoOpTextExtractor
from .registry import PLUGIN_REGISTRY, register_plugin
from .one_shot import create_schema_based_one_shot_extraction_plugin
