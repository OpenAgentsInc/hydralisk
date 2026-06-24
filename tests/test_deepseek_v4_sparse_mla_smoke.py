from __future__ import annotations

import json
from pathlib import Path

from hydralisk.admission.deepseek_v4_sparse_mla_smoke import (
    SCHEMA,
    main,
    render_markdown,
    run_smoke,
    write_smoke,
)


def test_sparse_mla_smoke_uses_public_safe_schema() -> None:
    smoke = run_smoke(
        heads=2,
        dim=4,
        page_size=4,
        sparse_capacity=2,
        seq_len=2,
        generated_at="2026-06-24T00:00:00+00:00",
    )

    assert smoke["schema"] == SCHEMA
    assert smoke["status"] == "ok"
    assert smoke["inputs"]["query"] == [1, 2, 4]
    assert smoke["inputs"]["swaKvCache"] == [1, 1, 4, 4]
    assert smoke["inputs"]["compressedKvCache"] == [1, 1, 4, 4]
    assert smoke["inputs"]["containsPrompts"] is False
    assert smoke["inputs"]["containsResponses"] is False
    assert smoke["inputs"]["loadsModelWeights"] is False
    assert smoke["result"]["outputShape"] == [1, 2, 4]
    assert smoke["result"]["finite"] is True
    assert smoke["result"]["nonzero"] is True
    assert smoke["result"]["checksum"]["count"] == 8
    assert smoke["decision"]["containerFallbackContractReady"] is True


def test_sparse_mla_smoke_markdown_is_public_safe() -> None:
    smoke = run_smoke(
        heads=1,
        dim=2,
        page_size=2,
        sparse_capacity=1,
        seq_len=1,
        generated_at="2026-06-24T00:00:00+00:00",
    )
    markdown = render_markdown(smoke)

    assert "# DeepSeek V4 sparse MLA fallback smoke" in markdown
    assert "Contains secrets: false" in markdown
    assert "Contains private prompts: false" in markdown
    assert "Container fallback contract ready: `True`" in markdown
    assert "bf16-compatible" in markdown


def test_sparse_mla_smoke_writes_json_and_markdown(tmp_path: Path) -> None:
    smoke = run_smoke(
        heads=1,
        dim=2,
        page_size=2,
        sparse_capacity=1,
        seq_len=1,
        generated_at="2026-06-24T00:00:00+00:00",
    )

    json_path, markdown_path = write_smoke(smoke, tmp_path)

    assert json.loads(json_path.read_text())["schema"] == SCHEMA
    assert markdown_path.read_text().startswith("# DeepSeek V4 sparse MLA")


def test_sparse_mla_smoke_cli_writes_outputs(tmp_path: Path, capsys) -> None:
    rc = main(
        [
            "--output-dir",
            str(tmp_path),
            "--heads",
            "1",
            "--dim",
            "2",
            "--page-size",
            "2",
            "--sparse-capacity",
            "1",
            "--seq-len",
            "1",
        ]
    )

    captured = capsys.readouterr()

    assert rc == 0
    assert "deepseek-v4-sparse-mla-fallback-smoke.json" in captured.out
    assert (tmp_path / "deepseek-v4-sparse-mla-fallback-smoke.json").exists()
    assert (tmp_path / "deepseek-v4-sparse-mla-fallback-smoke.md").exists()
