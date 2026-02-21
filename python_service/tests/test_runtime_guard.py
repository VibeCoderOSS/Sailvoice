from app import runtime_guard


def test_runtime_guard_missing_package(monkeypatch):
    def _raise_not_found(_name: str):
        raise runtime_guard.PackageNotFoundError

    monkeypatch.setattr(runtime_guard, 'distribution', _raise_not_found)
    status = runtime_guard.detect_runtime_status()

    assert status.installed is False
    assert status.compatible is False
    assert status.supportsQwen3Tts is False

