from __future__ import annotations

import subprocess
import sys


def test_public_primitives_and_evaluate_import_clean_process() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from dspy.primitives import Module; from dspy.evaluate.evaluator import Evaluate; "
            "print(Module.__name__, Evaluate.__name__)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "Module Evaluate"
