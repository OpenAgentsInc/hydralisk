from __future__ import annotations

import pytest

import hydralisk.serve.smoke as smoke_module


def test_smoke_cli_reads_bearer_token_from_env(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    async def fake_run_smoke(base_url: str, bearer_token: str, model: str):
        assert base_url == "http://localhost:8080"
        assert bearer_token == "secret-from-env"
        assert model == "glm-5.2-reap-504b-g4"
        return {"passed": True, "completionRunRef": "hydralisk-run-test"}

    monkeypatch.setenv("HYDRALISK_BEARER_TOKEN", "secret-from-env")
    monkeypatch.setattr(smoke_module, "run_smoke", fake_run_smoke)
    monkeypatch.setattr(
        "sys.argv",
        [
            "hydralisk-smoke",
            "--base-url",
            "http://localhost:8080",
            "--model",
            "glm-5.2-reap-504b-g4",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        smoke_module.main()

    assert exc_info.value.code == 0
    assert "hydralisk-run-test" in capsys.readouterr().out
