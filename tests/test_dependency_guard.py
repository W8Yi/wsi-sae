from __future__ import annotations

import json
import subprocess
import sys

from ._helpers import SRC


def test_base_package_import_does_not_pull_slide_backends():
    code = (
        "import json, sys; "
        f"sys.path.insert(0, {str(SRC)!r}); "
        "import wsi_sae; "
        "print(json.dumps({k: (k in sys.modules) for k in ['openslide', 'tifffile', 'PIL']}))"
    )
    result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, check=True)
    payload = json.loads(result.stdout.strip())
    assert payload == {"openslide": False, "tifffile": False, "PIL": False}

