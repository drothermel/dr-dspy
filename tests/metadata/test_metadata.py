import re

import dspy.__metadata__ as metadata


def test_metadata():
    assert metadata.__name__ == "dspy"
    assert re.match(r"\d+\.\d+\.\d+", metadata.__version__)
    assert metadata.__author__ == "Omar Khattab"
    assert metadata.__author_email__ == "okhattab@stanford.edu"
    assert metadata.__description__ == "DSPy"
