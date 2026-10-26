"""Private application tests: only synthetic media in pytest temporary roots."""
import struct
from uuid import uuid4

import numpy as np
import pytest
import soundfile as sf

from nightlab.services import media
from nightlab.services.records import RecordService
from nightlab.storage.repository import Repository, utc_now


def write_audio(path, subtype="PCM_16", seconds=1, channels=1, rate=16000):
    samples = np.linspace(-.7, .7, int(seconds * rate), dtype=np.float32)
    if channels == 2:
        samples = np.column_stack([samples, samples * .5])
    sf.write(path, samples, rate, subtype=subtype, format="WAV")
    return samples


def imported(tmp_path):
    service = RecordService(tmp_path / "private")
    stage = service.staging_dir / "upload"
    write_audio(stage)
    return service, service.import_audio(stage, "../../personal.wav", consent=True)


def job(record_id, status="running"):
    return dict(id=str(uuid4()), record_id=record_id, model_id="test_model", status=status, progress=0,
                processed_windows=0, total_windows=2, message="", error=None, created_at=utc_now(), finished_at=None)


def event(record_id, source="user", **extra):
    return dict(record_id=record_id, start_seconds=.1, end_seconds=.2, label="test", source=source,
                score=.8, model_id="test_model", **extra)


@pytest.mark.parametrize("subtype", ["PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE"])
@pytest.mark.parametrize("format", ["WAV", "WAVEX", "RF64"])
def test_real_formats(tmp_path, subtype, format):
    path = tmp_path / "not_an_extension"
    sf.write(path, np.zeros((800, 2)), 8000, subtype=subtype, format=format)
    info = media.audio_info_validate(path)
    assert info["frames"] == 800 and info["channels"] == 2
    assert info["format"] == format and len(info["sha256"]) == 64


def test_invalid_audio_and_truncated_container(tmp_path):
    path = tmp_path / "a.wav"
    path.write_bytes(b"not wav")
    with pytest.raises(ValueError):
        media.audio_info_validate(path)
    write_audio(path)
    good = path.read_bytes()
    for bad in (good[:-1], good + b"x", good[:40], good[:4] + struct.pack("<I", len(good) + 100) + good[8:]):
        path.write_bytes(bad)
        with pytest.raises(ValueError):
            media.audio_info_validate(path)
    # Header adjusted after truncation must still reject a non-frame-aligned chunk.
    truncated = bytearray(good[:-1])
    truncated[4:8] = struct.pack("<I", len(truncated) - 8)
    path.write_bytes(truncated)
    with pytest.raises(ValueError):
        media.audio_info_validate(path)


@pytest.mark.parametrize("samples", [np.array([0, np.nan]), np.array([np.inf, 0]), np.array([1.01, 0])])
def test_rejects_invalid_float_samples(tmp_path, samples):
    path = tmp_path / "a.wav"
    sf.write(path, samples, 16000, subtype="FLOAT")
    with pytest.raises(ValueError, match="非有限|超出"):
        media.audio_info_validate(path)


@pytest.mark.parametrize("rate,channels,subtype", [(4000, 1, "PCM_16"), (192000, 1, "PCM_16"), (16000, 3, "PCM_16"), (16000, 1, "PCM_U8")])
def test_rejects_unsupported_media(tmp_path, rate, channels, subtype):
    path = tmp_path / "a.wav"
    sf.write(path, np.zeros((10, channels)), rate, subtype=subtype)
    with pytest.raises(ValueError):
        media.audio_info_validate(path)


def test_empty_size_duration_and_cancellation_limits(tmp_path, monkeypatch):
    path = tmp_path / "a.wav"
    sf.write(path, np.zeros(0), 16000)
    with pytest.raises(ValueError):
        media.audio_info_validate(path)
    write_audio(path)
    monkeypatch.setattr(media, "MAX_BYTES", 10)
    with pytest.raises(ValueError, match="4GiB"):
        media.audio_info_validate(path)
    monkeypatch.setattr(media, "MAX_BYTES", 4 * 1024**3)
    monkeypatch.setattr(media, "MAX_SECONDS", .5)
    with pytest.raises(ValueError, match="24小时"):
        media.audio_info_validate(path)
    monkeypatch.setattr(media, "MAX_SECONDS", 86400)
    with pytest.raises(InterruptedError):
        media.audio_info_validate(path, stop_requested=lambda: True)


def test_window_waveform_grid_and_bounded_stream(tmp_path, monkeypatch):
    path = tmp_path / "a.wav"
    samples = write_audio(path, subtype="FLOAT", channels=2)
    output, rate = media.read_mono_window(path, .125, .25)
    assert output.dtype == np.float32 and rate == 16000
    np.testing.assert_allclose(output, samples[2000:6000].mean(axis=1), atol=1e-7)
    monkeypatch.setattr(media, "BLOCK_FRAMES", 7)
    result = media.waveform(path, points=10, start=.125, end=.625)
    assert len(result["points"]) == 10
    for index, point in enumerate(result["points"]):
        subset = samples[2000 + index * 800:2800 + index * 800].mean(axis=1)
        assert point["min"] == pytest.approx(subset.min())
        assert point["max"] == pytest.approx(subset.max())
        assert point["time_seconds"] == pytest.approx((2400 + index * 800) / rate)
    for start, duration in [(-1, 1), (0, 0), (0, 2), (float("nan"), 1), (.1, 1e-10)]:
        with pytest.raises(ValueError):
            media.read_mono_window(path, start, duration)
    with pytest.raises(ValueError):
        media.waveform(path, points=4001)


def test_spectrogram_png_and_demo(tmp_path):
    path = tmp_path / "demo.wav"
    demo = media.deterministic_demo(path)
    info = media.audio_info_validate(path)
    assert info["duration_seconds"] == 90 and info["subtype"] == "PCM_16"
    assert len(demo["events"]) == 3 and all(e["source"] == "demo" for e in demo["events"])
    assert demo["wearable"]["summary"]["synthetic"] is True
    assert media.spectrogram_png(path, start=14, duration=2).startswith(b"\x89PNG\r\n\x1a\n")
    with pytest.raises(ValueError):
        media.spectrogram_png(path, duration=61)
    other = tmp_path / "other.wav"
    media.deterministic_demo(other)
    assert path.read_bytes() == other.read_bytes()


def test_persistence_source_and_atomic_analysis(tmp_path):
    service, record = imported(tmp_path)
    repo, rid = service.repo, record["id"]
    user = repo.add_event(event(rid))
    assert user["score"] is None and user["model_id"] is None
    old = repo.add_event(event(rid, "model"))
    analysis = repo.create_job(job(rid))
    repo.set_wearable(rid, {"summary": {"synthetic": True}})
    with pytest.raises(ValueError):
        repo.delete_event(rid, old["id"])
    broken = event(rid, "model")
    broken["score"] = float("nan")
    with pytest.raises(ValueError):
        repo.complete_analysis(analysis["id"], [broken])
    assert {e["id"] for e in repo.get_events(rid)} == {user["id"], old["id"]}
    assert repo.get_job(analysis["id"])["status"] == "running"
    completed = repo.complete_analysis(analysis["id"], [event(rid, "model")])
    assert completed["progress"] == 1 and completed["finished_at"]
    fresh = Repository(service.data_dir)
    assert fresh.get_record(rid) == record
    assert fresh.get_wearable(rid)["summary"]["synthetic"]
    assert len(fresh.get_events(rid)) == 2 and user in fresh.get_events(rid)
    assert old not in fresh.get_events(rid)
    repo.delete_event(rid, user["id"])
    with pytest.raises(KeyError):
        repo.get_record(str(uuid4()))


def test_import_rollback_consent_timezone_and_paths(tmp_path, monkeypatch):
    service = RecordService(tmp_path / "private")
    outside = tmp_path / "outside.wav"
    write_audio(outside)
    with pytest.raises(ValueError):
        service.import_audio(outside, "outside", consent=True)
    assert outside.exists()
    for kwargs in [{}, {"consent": True, "start_time": "2026-10-26T01:00:00"}]:
        stage = service.staging_dir / "stage"
        write_audio(stage)
        with pytest.raises(ValueError):
            service.import_audio(stage, "test", **kwargs)
        assert not stage.exists()
    stage = service.staging_dir / "stage"
    write_audio(stage)
    def fail(*args):
        raise RuntimeError("test disk db failure")
    monkeypatch.setattr(service.repo, "create_record", fail)
    with pytest.raises(RuntimeError):
        service.import_audio(stage, "test", consent=True)
    assert not stage.exists() and not list(service.records_dir.iterdir())
    assert service.repo.list_records() == []
    with pytest.raises(ValueError):
        service.audio_path("../../outside.wav")


def test_deletion_physical_and_relations_and_active_job(tmp_path, monkeypatch):
    service, record = imported(tmp_path)
    rid, repo = record["id"], service.repo
    root = service.audio_path(rid).parent
    (root / "derived").mkdir()
    (root / "derived" / "cache").write_text("synthetic")
    repo.add_event(event(rid))
    repo.set_wearable(rid, {"summary": {}})
    active = repo.create_job(job(rid, "queued"))
    with pytest.raises(ValueError):
        service.delete_record(rid)
    with pytest.raises(RuntimeError):
        service.delete_record(rid, confirmed=True)
    assert root.exists() and repo.get_record(rid)
    repo.update_job(active["id"], status="cancelled")
    original_remove = service._remove_tree
    def fail(*args):
        raise OSError("test IO failure")
    monkeypatch.setattr(service, "_remove_tree", fail)
    with pytest.raises(OSError):
        service.delete_record(rid, confirmed=True)
    assert repo.get_record(rid) and root.exists()
    monkeypatch.setattr(service, "_remove_tree", original_remove)
    assert service.delete_record(rid, confirmed=True)["deleted"]
    assert not root.exists() and not repo.list_records()
    with repo._connection() as db:
        for table in ("records", "jobs", "events", "wearable"):
            assert db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0


def test_symlink_escape_rejected_without_deleting_target(tmp_path):
    service, record = imported(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "keep"
    target.write_text("preserve")
    link = service.audio_path(record["id"]).parent / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Windows user lacks symlink privilege")
    with pytest.raises(ValueError):
        service.delete_record(record["id"], confirmed=True)
    assert target.read_text() == "preserve" and service.repo.get_record(record["id"])
