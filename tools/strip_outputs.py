"""Git clean filter for notebooks: reads a notebook on stdin, writes it without outputs and execution counts on stdout.

The notebook in the working folder keeps its outputs; only what is committed is stripped. Set it up once per clone:

    git config filter.nbclean.clean "python tools/strip_outputs.py"
    git config filter.nbclean.required true
"""
import json
import sys

nb = json.loads(sys.stdin.buffer.read().decode("utf8"))
for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
sys.stdout.buffer.write((json.dumps(nb, indent=1, ensure_ascii=False) + "\n").encode("utf8"))
