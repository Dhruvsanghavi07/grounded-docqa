import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from docqa.chunking import load_corpus  # noqa: E402
from docqa.index import HybridIndex  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures" / "corpus"


@pytest.fixture(scope="session")
def chunks():
    return load_corpus(FIXTURES)


@pytest.fixture(scope="session")
def index(chunks):
    return HybridIndex().build(chunks)
