"""Every ```python block in README.md and docs/*.md must run. Blocks in one file run top to bottom in one
namespace, so a later block may use names an earlier one defined."""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FILES = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
BLOCK = re.compile(r"^```python\n(.*?)^```", re.S | re.M)


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_python_blocks_in_docs_run(path):
    blocks = BLOCK.findall(path.read_text())
    assert blocks or path.name == "api.md", f"{path.name} has no runnable example"
    ns: dict = {"__name__": f"doc_{path.stem}"}
    for i, code in enumerate(blocks, 1):
        exec(compile(code, f"{path.name}[block {i}]", "exec"), ns)  # noqa: S102


def test_docs_exist():
    assert {p.name for p in FILES} >= {"README.md", "usage.md", "api.md"}
