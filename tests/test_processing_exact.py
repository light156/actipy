"""Exactness of the processing shortcuts against the code they replace."""

import numpy as np
import pandas as pd
import pytest

from actipy import processing as P


def _index(values, unit="ns", tz=None):
    index = pd.DatetimeIndex(np.asarray(values, dtype=np.int64).view("datetime64[ns]"))
    if unit != "ns":
        index = index.as_unit(unit) if hasattr(index, "as_unit") else index
    return index.tz_localize("UTC").tz_convert(tz) if tz else index


@pytest.mark.parametrize("unit", ["ns", "ms", "us"])
@pytest.mark.parametrize("tz", [None, "Europe/London"])
def test_nearest_indexer_matches_pandas_get_indexer(unit, tz):
    rng = np.random.default_rng(0)
    tolerance = pd.Timedelta("1s")
    for _ in range(50):
        # ~100 Hz with jitter, gaps of a few samples up to minutes, exact hits
        steps = rng.choice([10, 10, 9, 11, 30, 2500, 120_000], size=400) * 1_000_000
        source = _index(1_700_000_000_000_000_000 + np.cumsum(steps), unit, tz)
        start = source[0] + pd.Timedelta(int(rng.integers(0, 5000)), "ms")
        target = pd.date_range(start, periods=int(rng.integers(1, 300)), freq="10ms")
        expected = source.get_indexer(target, method="nearest", tolerance=tolerance, limit=1)
        np.testing.assert_array_equal(P._nearest_indexer(source, target, tolerance), expected)


@pytest.mark.parametrize("with_temperature", [False, True])
def test_wls_params_match_statsmodels(with_temperature):
    sm = pytest.importorskip("statsmodels.api")
    rng = np.random.default_rng(1)
    for _ in range(20):
        n = int(rng.integers(50, 2000))
        inp = rng.normal(size=n)
        if with_temperature:
            inp = np.column_stack((inp, rng.normal(25, 3, size=n)))
        out = rng.normal(size=n)
        weights = np.maximum(1 - rng.random(n) / 0.9, 0)
        expected = sm.WLS(out, sm.add_constant(inp, prepend=True, has_constant="add"), weights=weights).fit().params
        exog = np.column_stack([np.ones(n), inp if inp.ndim == 2 else inp[:, None]])
        assert P._wls_params(out, exog, weights).tobytes() == expected.tobytes()
