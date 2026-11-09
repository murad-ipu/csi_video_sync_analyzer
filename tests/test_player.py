"""Tests for the player's signal processing, web API, and sync estimate."""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

import make_demo_session
import player


def test_select_subcarriers_drops_zero_columns_but_keeps_rare_zeros():
    amplitude = np.ones((100, 4))
    amplitude[:, 1] = 0
    amplitude[:5, 2] = 0  # a few zero readings do not make a null subcarrier
    assert player.select_subcarriers(amplitude).tolist() == [0, 2, 3]


def test_normalize_removes_per_packet_scale():
    base = np.array([[10.0, 20.0, 30.0]])
    scaled = np.concatenate((base, base * 2.5))
    normalized = player.normalize_per_frame(scaled)
    assert normalized[0] == pytest.approx(normalized[1])
    assert normalized[0].mean() == pytest.approx(1.0)


def test_normalize_leaves_an_all_zero_frame_at_zero():
    assert player.normalize_per_frame(np.zeros((1, 3))).tolist() == [[0.0, 0.0, 0.0]]


def test_motion_signal_matches_direct_variance():
    rng = np.random.default_rng(1)
    t = np.arange(0, 5, 0.01)
    data = rng.normal(1, 0.2, (t.size, 8))
    result = player.motion_signal(t, data, window_s=0.5)
    i = 200
    mask = (t >= t[i] - 0.25) & (t <= t[i] + 0.25)
    assert result[i] == pytest.approx(data[mask].var(axis=0).mean(), rel=1e-5)


def test_motion_peak_is_not_delayed():
    t = np.arange(0, 10, 0.01)
    data = np.ones((t.size, 4))
    burst = (t > 4.9) & (t < 5.1)
    data[burst] += np.random.default_rng(2).normal(0, 0.5, (burst.sum(), 4))
    result = player.motion_signal(t, data, window_s=0.5)
    # A burst shorter than the window gives a flat top, so use the top's center.
    top = t[result >= 0.9 * result.max()]
    assert (top.min() + top.max()) / 2 == pytest.approx(5.0, abs=0.02)


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    folder = tmp_path_factory.mktemp("demo") / "s1"
    make_demo_session.make_session(folder, 30.0, 120.0, width=320, height=240)
    return folder, player.load_session(folder)


@pytest.fixture(scope="module")
def client(session):
    return TestClient(player.create_app(session[1]))


def test_meta_drops_null_bins_and_array_sizes_match(client):
    meta = client.get("/api/meta").json()
    assert len(meta["kept_bins"]) == 52 and 32 not in meta["kept_bins"]
    amp = np.frombuffer(client.get("/api/array/amp_norm").content, "<f4")
    assert amp.size == meta["n_csi"] * 52
    assert np.frombuffer(client.get("/api/array/frame_t").content, "<f8").size == meta["n_frames"]
    assert np.frombuffer(client.get("/api/array/chart_motion").content, "<f4").size < meta["n_csi"]


def test_unknown_array_is_404(client):
    assert client.get("/api/array/secret").status_code == 404


def test_video_supports_range_requests(client):
    response = client.get("/video.mp4", headers={"Range": "bytes=0-99"})
    assert response.status_code == 206
    assert len(response.content) == 100


def test_index_page_and_vendored_chart_library_are_served(client):
    assert "CSI Video Player" in client.get("/").text
    assert client.get("/vendor/uPlot.iife.min.js").status_code == 200


def test_sync_offset_is_saved_and_other_fields_are_kept(client, session):
    folder, _ = session
    assert client.post("/api/sync_offset", json={"sync_offset_ms": -37.5}).status_code == 200
    saved = json.loads((folder / "session.json").read_text())
    assert saved["sync_offset_ms"] == -37.5
    assert saved["status"] == "synthetic"
    assert client.get("/api/meta").json()["sync_offset_ms"] == -37.5
    client.post("/api/sync_offset", json={"sync_offset_ms": 0})


@pytest.mark.parametrize("bad", [99999, "abc", None])
def test_sync_offset_rejects_bad_values(client, bad):
    assert client.post("/api/sync_offset", json={"sync_offset_ms": bad}).status_code == 422


def test_estimate_sync_finds_the_known_offset_from_real_video(session):
    result = player.estimate_sync(session[1])
    assert result["confident"], result
    assert result["offset_ms"] == pytest.approx(120.0, abs=40.0)


def test_unfinished_session_points_to_recover(tmp_path):
    (tmp_path / "csi_stream.bin").write_bytes(b"x")
    with pytest.raises(SystemExit, match="--recover"):
        player.load_session(tmp_path)
