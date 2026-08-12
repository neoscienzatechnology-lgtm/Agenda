import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

FIXTURES = ROOT / "tests" / "fixtures"


@pytest.fixture(scope="session")
def scenes():
    """Cache de cenas sintéticas — renderizá-las é caro."""
    from app.synth import generator as G

    cache: dict[tuple, object] = {}

    def get(**kwargs):
        key = tuple(sorted(kwargs.items()))
        if key not in cache:
            single = kwargs.pop("single", False)
            if single:
                spec = G.single_foot_scene(**kwargs)
            else:
                spec = G.bilateral_scene(**kwargs)
            cache[key] = G.render_scene(spec)
        return cache[key]

    return get


@pytest.fixture(scope="session")
def parity_fixture():
    import json

    return json.loads((FIXTURES / "measurement_parity.json").read_text(encoding="utf-8"))
