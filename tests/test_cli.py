"""The public CLI uses only local data and retained experiment commands."""
import json

import numpy as np
import pytest

from tshybrid.cli import main, prepare_data


def test_help_lists_only_supported_commands(capsys):
    with pytest.raises(SystemExit) as result:
        main(["--help"])
    assert result.value.code == 0
    help_text = capsys.readouterr().out
    for name in ("run", "inspect", "report", "combine", "prepare-data", "prefetch", "lock-models"):
        assert name in help_text
    for name in ("compare", "import-hn-reference", "hybrid-report"):
        assert name not in help_text


@pytest.mark.parametrize("command", ["compare", "import-hn-reference", "hybrid-report"])
def test_removed_commands_are_rejected(command):
    with pytest.raises(SystemExit) as result:
        main([command])
    assert result.value.code == 2


def test_prepare_data_copies_from_an_arbitrary_folder(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "data"
    source.mkdir()
    original = np.arange(120, dtype=float).reshape(60, 2)
    np.save(source / "ETTm1.npy", original)
    prepare_data(source, destination, ["ETTm1"])
    np.testing.assert_array_equal(np.load(destination / "ETTm1.npy"), original)
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["ETTm1"]["source"] == str((source / "ETTm1.npy").resolve())
    assert manifest["ETTm1"]["sha256"]
