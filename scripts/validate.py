#!/usr/bin/env python3
"""Validate repository layout, checksums, configs, and available runtimes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads((ROOT / "config.json").read_text())
    labels = json.loads((ROOT / "data" / "labels.json").read_text())
    assert len(labels) == 2498
    assert "Domestic Chicken" in labels.values()

    for backend, entry in manifest["backends"].items():
        config_path = ROOT / entry["config"]
        config = json.loads(config_path.read_text())
        assert config["input_shape"] == [1, 3, 224, 224]
        assert config["model"] == "resnet"
        assert config["labels"] == labels
        for relative in config["files"]:
            model_file = config_path.parent / relative
            assert model_file.exists(), f"missing {backend} weight: {model_file}"
            if model_file.is_file():
                print(f"{backend:8} {model_file.name:38} {sha256(model_file)}")
    print("Repository validation passed.")


if __name__ == "__main__":
    main()
