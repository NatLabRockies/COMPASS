"""COMPASS Geothermal Resource Management Plan plugin"""

import importlib.resources
from pathlib import Path

from compass.plugin import create_schema_based_one_shot_extraction_plugin, OutputColumn
from compass.utilities import doc_infos_to_db, save_db


COMPASSGeoRMPExtractor = create_schema_based_one_shot_extraction_plugin(
    importlib.resources.files("compass.extraction.rmp") / "plugin_config.yaml",
    tech="geo_rmp",
)

_out_cols = getattr(COMPASSGeoRMPExtractor, "OUTPUT_COLUMNS", [])
if not any(col.name == "restriction_level" for col in _out_cols):
    _out_cols.append(OutputColumn("restriction_level"))
if not any(col.name == "document_name" for col in _out_cols):
    _out_cols.append(OutputColumn("document_name"))


@classmethod
def _save_structured_data(cls, doc_infos, out_dir):
    """Save RMP extraction results, adding document_name from source path"""
    output_cols = getattr(cls, "OUTPUT_COLUMNS", [])
    db, num_docs_found = doc_infos_to_db(doc_infos, output_cols)

    if not db.empty:
        db["document_name"] = db["source"].apply(
            lambda src: Path(src).name if isinstance(src, str) and src else None
        )

    save_db(db, out_dir, output_cols)
    return num_docs_found


COMPASSGeoRMPExtractor.save_structured_data = _save_structured_data
