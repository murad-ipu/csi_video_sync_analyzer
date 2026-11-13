"""Tests for analyze.py and espectre_bridge.py: ESPectre reuse, saved results, and the cache."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

import analyze
import espectre_bridge
import make_demo_session
import player

ROOT = Path(__file__).resolve().parent.parent
# Real recordings are not in git. Tests that use them are skipped when none exist.
REAL_SESSIONS = sorted(path.parent for path in (ROOT / "sessions").glob("**/csi.npz")
                       if (path.parent / "csi_meta.json").exists()
                       and json.loads((path.parent / "csi_meta.json").read_text()).get("sensing"))


def old_amplitude(iq: np.ndarray) -> np.ndarray:
    """The amplitude formula record.py used before ESPectre's CSIPacket replaced it."""
    pairs = np.asarray(iq).reshape(iq.shape[0], -1, 2).astype(np.float32)
    return np.hypot(pairs[..., 0], pairs[..., 1])


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    folder = tmp_path_factory.mktemp("analyze") / "s1"
    make_demo_session.make_session(folder, 30.0, 0.0, width=320, height=240)
    manifest = analyze.run_analysis(folder, log=lambda *_: None)
    return folder, manifest


# --- Step 2: amplitude and phase from ESPectre --------------------------------------

def test_espectre_amplitude_equals_the_old_formula_on_the_demo_session(session):
    folder, _ = session
    iq = np.load(folder / "csi.npz")["csi_data"]
    amplitude, phase = espectre_bridge.amplitude_phase(iq)
    assert np.allclose(amplitude, old_amplitude(iq), rtol=1e-6, atol=1e-5)
    # Phase is the angle of I + jQ, as in ESPectre's parser.
    q, i = iq[:, 0::2].astype(np.float32), iq[:, 1::2].astype(np.float32)
    assert np.allclose(phase, np.arctan2(q, i), atol=1e-6)


@pytest.mark.skipif(not REAL_SESSIONS, reason="no real recording in sessions/")
@pytest.mark.parametrize("folder", REAL_SESSIONS, ids=lambda path: path.name)
def test_espectre_amplitude_equals_the_stored_amplitude_of_real_recordings(folder):
    """Recordings made with the old formula must give the same values through ESPectre."""
    stored = np.load(folder / "csi.npz")
    amplitude, _ = espectre_bridge.amplitude_phase(stored["csi_data"])
    assert np.allclose(amplitude, stored["amplitude"], rtol=1e-6, atol=1e-5)
    assert np.allclose(amplitude, old_amplitude(stored["csi_data"]), rtol=1e-6, atol=1e-5)


# --- Step 3: ESPectre detector replay --------------------------------------------------

def assert_matches_espectre_replay_tool(folder: Path, results: dict) -> None:
    """Our states on scored evaluations equal performance_report.replay_idle_stream."""
    import config
    from tools.lib import performance_report

    sensing = json.loads((folder / "csi_meta.json").read_text()).get("sensing")
    board = espectre_bridge.board_settings(sensing)
    packets, _ = espectre_bridge.load_packets(folder / "csi.npz", board["csi_target_pps"])
    for name in espectre_bridge.DETECTORS:
        detector, warmup, _ = espectre_bridge.build_detector(name, packets, board)
        expected = performance_report.replay_idle_stream(detector, packets, config.DEFAULT_SUBCARRIERS, warmup)
        ours = results[name]
        assert len(expected) > 0
        assert ours["raw_motion"][ours["ready"]].tolist() == expected, name


def test_detector_replay_matches_espectre_replay_tool_on_the_demo_session(session):
    folder, _ = session
    results = espectre_bridge.replay_detectors(folder / "csi.npz", json.loads((folder / "csi_meta.json").read_text())["sensing"])
    assert_matches_espectre_replay_tool(folder, results)


@pytest.mark.skipif(not REAL_SESSIONS, reason="no real recording in sessions/")
def test_detector_replay_matches_espectre_replay_tool_on_a_real_recording():
    folder = REAL_SESSIONS[0]
    results = espectre_bridge.replay_detectors(folder / "csi.npz", json.loads((folder / "csi_meta.json").read_text())["sensing"])
    assert_matches_espectre_replay_tool(folder, results)


def test_detector_results_use_the_board_settings_and_the_evaluation_grid(session):
    folder, manifest = session
    data = analyze.load_results(folder, np.load(folder / "csi.npz"))
    csi_t_ns = np.load(folder / "csi.npz")["csi_t_ns"]
    for name in espectre_bridge.DETECTORS:
        result, info = data["detectors"][name], manifest["detectors"][name]
        assert info["motion_on_hits"] == 4 and info["motion_off_hits"] == 3
        assert info["n_ready"] > 0
        # One evaluation every 250 ms, on the CSI time of the packet that triggered it.
        spacing_s = np.diff(result["t_ns"]) / 1e9
        assert np.median(spacing_s) == pytest.approx(0.25, abs=0.02)
        assert (result["t_ns"] == csi_t_ns[result["row"]]).all()
        assert set(np.unique(result["state"])) <= {0, 1, 2}
        assert ((result["state"] == 0) == ~result["ready"]).all()
        assert ((result["score"] >= 0) & (result["score"] <= 1)).all()
    # The demo board ran High-Accuracy, so its threshold comes from /sensing.
    assert manifest["detectors"]["high_accuracy"]["threshold_source"] == "board (/sensing)"
    assert manifest["detectors"]["lightweight"]["threshold_source"] != "board (/sensing)"


def test_both_detectors_share_one_evaluation_grid(session):
    folder, _ = session
    data = analyze.load_results(folder, np.load(folder / "csi.npz"))
    assert np.array_equal(data["detectors"]["lightweight"]["t_ns"], data["detectors"]["high_accuracy"]["t_ns"])


# --- Step 4: compute once, store, reuse ------------------------------------------------

def derived_mtimes(folder: Path) -> dict:
    return {path.name: path.stat().st_mtime_ns for path in (folder / "derived").iterdir()}


def test_manifest_records_inputs_code_and_espectre_commit(session):
    folder, manifest = session
    fingerprint = manifest["fingerprint"]
    assert set(fingerprint["inputs"]) == {"csi.npz", "frames.csv", "csi_meta.json"}
    assert len(fingerprint["code_sha256"]) == 64
    assert fingerprint["espectre"]["commit"] != ""
    assert fingerprint["parameters"]["espectre"]["evaluation_interval_ms"] == 250
    assert manifest["frequency_default"]["method"] == "mean"
    assert set(manifest["files"]) >= {"player_signals.npz", "espectre_lightweight.npz", "espectre_high_accuracy.npz"}


def test_a_second_run_reuses_the_saved_results(session):
    folder, _ = session
    before = derived_mtimes(folder)
    manifest, computed = analyze.ensure_analysis(folder, log=lambda *_: None)
    assert not computed
    assert derived_mtimes(folder) == before


def test_the_player_only_loads_saved_results(session, monkeypatch):
    folder, _ = session

    def fail(*args, **kwargs):
        raise AssertionError("the player recomputed although the results were up to date")

    monkeypatch.setattr(analyze, "run_analysis", fail)
    monkeypatch.setattr(espectre_bridge, "replay_detectors", fail)
    data = player.load_session(folder)
    assert data["meta"]["espectre"]["detectors"]["high_accuracy"]["n_ready"] > 0


def test_the_default_frequency_view_is_already_cached(session):
    import frequency
    folder, manifest = session
    assert frequency.cache_path(folder, manifest["frequency_default"]["key"]).is_file()


def test_a_changed_input_triggers_a_recompute(tmp_path):
    folder = tmp_path / "s"
    make_demo_session.make_session(folder, 12.0, 0.0, width=160, height=120)
    analyze.run_analysis(folder, log=lambda *_: None)
    meta = json.loads((folder / "csi_meta.json").read_text())
    meta["sensing"]["motion_on_hits"] = 2
    (folder / "csi_meta.json").write_text(json.dumps(meta))
    assert not analyze.is_up_to_date(folder)
    manifest, computed = analyze.ensure_analysis(folder, log=lambda *_: None)
    assert computed and manifest["detectors"]["high_accuracy"]["motion_on_hits"] == 2
    assert analyze.ensure_analysis(folder, recompute=True, log=lambda *_: None)[1]


def test_another_espectre_commit_triggers_a_recompute(session, monkeypatch):
    folder, _ = session
    real_git = analyze._git

    def other_commit(root, *args):
        return "0" * 40 if args == ("rev-parse", "HEAD") else real_git(root, *args)

    monkeypatch.setattr(analyze, "_git", other_commit)
    assert not analyze.is_up_to_date(folder)


def test_a_missing_result_file_triggers_a_recompute(tmp_path):
    folder = tmp_path / "s"
    make_demo_session.make_session(folder, 12.0, 0.0, width=160, height=120)
    analyze.run_analysis(folder, log=lambda *_: None)
    os.remove(folder / "derived" / "espectre_lightweight.npz")
    assert not analyze.is_up_to_date(folder)


def test_old_recordings_get_their_values_in_the_derived_folder(tmp_path):
    """A csi.npz written before phase was stored is left untouched."""
    folder = tmp_path / "s"
    make_demo_session.make_session(folder, 12.0, 0.0, width=160, height=120)
    with np.load(folder / "csi.npz") as stored:
        arrays = {name: stored[name] for name in stored.files if name not in ("phase", "csi_target_pps")}
    np.savez(folder / "csi.npz", **arrays)
    before = (folder / "csi.npz").read_bytes()
    manifest = analyze.run_analysis(folder, log=lambda *_: None)
    assert manifest["csi_values_source"] == "derived/csi_values.npz"
    assert (folder / "csi.npz").read_bytes() == before
    data = analyze.load_results(folder, np.load(folder / "csi.npz"))
    assert data["phase"].shape == data["amplitude"].shape
