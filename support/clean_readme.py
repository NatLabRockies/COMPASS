"""Clean README before packaging"""

import sys
import pathlib


PYPI_DISALLOWED_RST = {"raw::", "<p", "</p", "<img", "---------"}
REMOVE_TEXT = ["&nbsp;"]


def _clean(fp):
    """Prep the README file for PyPi distribution"""
    readme = []
    with pathlib.Path(fp).open(encoding="utf-8") as f:
        for line in f:
            if any(substr in line for substr in PYPI_DISALLOWED_RST):
                continue
            readme.append(line)

    readme = "".join(readme)
    for substr in REMOVE_TEXT:
        readme = readme.replace(substr, "")

    readme = readme.lstrip()
    pathlib.Path(fp).write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    _clean(*sys.argv[1:])
