"""LSL marker stream: markers sent by the experiment arrive at an LSL inlet. Skipped without pylsl."""
import pytest

pylsl = pytest.importorskip("pylsl")

from sfu.hardware import make_markers  # noqa: E402


def test_markers_reach_lsl_inlet():
    m = make_markers("lsl", lsl_name="sfu_test_markers")
    streams = pylsl.resolve_byprop("name", "sfu_test_markers", timeout=5)
    assert streams and streams[0].type() == "Markers"
    inlet = pylsl.StreamInlet(streams[0])
    inlet.open_stream(timeout=5)
    sent = ["block_start block=1 type=main cond=narrow", "stim_on trial=5 cond=narrow sf=2.000", "stim_off"]
    t0 = pylsl.local_clock()
    for s in sent:
        event, _, info = s.partition(" ")
        m.send(event, None, info)
    got = []
    while len(got) < len(sent):
        sample, ts = inlet.pull_sample(timeout=5)
        assert sample is not None, got
        got.append((sample[0], ts))
    assert [g[0] for g in got] == sent
    assert all(t0 - 0.01 <= ts <= pylsl.local_clock() for _, ts in got)
    m.close()
