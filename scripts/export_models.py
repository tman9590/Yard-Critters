#!/usr/bin/env python3
"""Export the SpeciesNet crop classifier to every Scrypted custom backend."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn


INPUT_SHAPE = (1, 3, 480, 480)
BACKENDS = ("onnx", "openvino", "coreml", "ncnn")
ARTIFACT_VERSION = "v1.0.2"


class ScryptedSpeciesNet(nn.Module):
    """Adapt Scrypted's NCHW [0,1] tensor to SpeciesNet's NHWC [0,1] input."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image.permute(0, 2, 3, 1))


def labels_from_taxonomy(path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        fields = line.split(";")
        common_name = fields[-1].strip()
        scientific = " ".join(part for part in fields[4:6] if part).strip()
        label = common_name or scientific or fields[1] or f"species-{index}"
        labels[str(index)] = label.title()
    return labels


def write_config(repo: Path, backend: str, labels: dict[str, str], files: list[str]) -> None:
    config = {
        "input_shape": list(INPUT_SHAPE),
        "model": "resnet",
        "files": files,
        "labels": labels,
    }
    destination = repo / "models" / backend / "config.json"
    destination.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def export_onnx(repo: Path, model: nn.Module, example: torch.Tensor) -> Path:
    destination = repo / "models" / "onnx" / f"yard-critters-{ARTIFACT_VERSION}.onnx"
    torch.onnx.export(
        model,
        example,
        destination,
        input_names=["input"],
        output_names=["out0"],
        opset_version=17,
        do_constant_folding=True,
    )
    return destination


def export_openvino(repo: Path, onnx_path: Path) -> list[str]:
    import openvino as ov

    model = ov.convert_model(onnx_path)
    destination = repo / "models" / "openvino" / f"yard-critters-{ARTIFACT_VERSION}.xml"
    ov.save_model(model, destination, compress_to_fp16=True)
    return [
        f"yard-critters-{ARTIFACT_VERSION}.xml",
        f"yard-critters-{ARTIFACT_VERSION}.bin",
    ]


def export_coreml(repo: Path, model: nn.Module, example: torch.Tensor) -> list[str]:
    import coremltools as ct

    traced = torch.jit.trace(model, example, strict=False)
    converted = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.ImageType(name="input", shape=INPUT_SHAPE, scale=1 / 255.0)],
        outputs=[ct.TensorType(name="out0")],
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=ct.precision.FLOAT16,
    )
    package = f"yard-critters-{ARTIFACT_VERSION}.mlpackage"
    destination = repo / "models" / "coreml" / package
    if destination.exists():
        shutil.rmtree(destination)
    converted.save(destination)
    return [
        f"{package}/Manifest.json",
        f"{package}/Data/com.apple.CoreML/model.mlmodel",
        f"{package}/Data/com.apple.CoreML/weights/weight.bin",
    ]


def export_ncnn(repo: Path, onnx_path: Path) -> list[str]:
    destination = repo / "models" / "ncnn"
    pnnx = Path(sys.executable).parent / "pnnx"
    subprocess.run(
        [str(pnnx), str(onnx_path.resolve()), f"inputshape={list(INPUT_SHAPE)}"],
        cwd=destination,
        check=True,
    )
    generated_stem = onnx_path.stem.replace("-", "_")
    candidates = list(onnx_path.parent.rglob(f"{generated_stem}.ncnn.param"))
    if not candidates:
        raise FileNotFoundError("pnnx did not create an NCNN parameter file")
    source_param = candidates[0]
    source_bin = source_param.with_suffix(".bin")
    target_param = destination / f"yard-critters-{ARTIFACT_VERSION}.ncnn.param"
    target_bin = destination / f"yard-critters-{ARTIFACT_VERSION}.ncnn.bin"
    if source_param != target_param:
        shutil.move(source_param, target_param)
    if source_bin != target_bin:
        shutil.move(source_bin, target_bin)
    # Scrypted's NCNN adapter extracts the fixed output name "out0".
    text = target_param.read_text(encoding="utf-8")
    lines = text.splitlines()
    if lines and not any(" out0" in line for line in lines):
        last = lines[-1].split()
        if len(last) >= 4:
            last[3] = "out0"
            lines[-1] = " ".join(last)
            target_param.write_text("\n".join(lines) + "\n", encoding="utf-8")
    for suffix in (
        ".pnnx.param",
        ".pnnx.bin",
        ".pnnx.onnx",
        ".pnnxsim.onnx",
        "_pnnx.py",
        "_ncnn.py",
    ):
        auxiliary = onnx_path.parent / f"{generated_stem}{suffix}"
        if auxiliary.exists():
            auxiliary.unlink()
    return [target_param.name, target_bin.name]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--backend", choices=(*BACKENDS, "all"), default="all")
    args = parser.parse_args()

    repo = args.repo.resolve()
    labels = labels_from_taxonomy(args.labels)
    (repo / "data" / "labels.json").write_text(
        json.dumps(labels, indent=2) + "\n", encoding="utf-8"
    )

    base = torch.load(args.source_model, map_location="cpu", weights_only=False).eval()
    model = ScryptedSpeciesNet(base).eval()
    example = torch.zeros(INPUT_SHAPE, dtype=torch.float32)

    requested = BACKENDS if args.backend == "all" else (args.backend,)
    onnx_path = (
        repo / "models" / "onnx" / f"yard-critters-{ARTIFACT_VERSION}.onnx"
    )
    if "onnx" in requested or any(x in requested for x in ("openvino", "ncnn")):
        onnx_path = export_onnx(repo, model, example)
        write_config(repo, "onnx", labels, [onnx_path.name])
    if "openvino" in requested:
        write_config(repo, "openvino", labels, export_openvino(repo, onnx_path))
    if "coreml" in requested:
        write_config(repo, "coreml", labels, export_coreml(repo, model, example))
    if "ncnn" in requested:
        write_config(repo, "ncnn", labels, export_ncnn(repo, onnx_path))


if __name__ == "__main__":
    main()
