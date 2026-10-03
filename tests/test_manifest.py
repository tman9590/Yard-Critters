import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_universal_manifest_has_every_scrypted_custom_backend():
    manifest = json.loads((ROOT / "config.json").read_text())
    assert set(manifest["backends"]) == {"coreml", "openvino", "onnx", "ncnn"}


def test_chicken_is_a_native_label():
    labels = json.loads((ROOT / "data" / "labels.json").read_text())
    assert "Domestic Chicken" in labels.values()
