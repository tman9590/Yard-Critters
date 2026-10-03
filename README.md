# Yard Critters

Yard Critters packages Google's **SpeciesNet v4.0.3a crop classifier** for Scrypted. It recognizes 2,498 wildlife and domestic-animal taxa—including **Domestic Chicken**, Red Junglefowl, Wild Turkey, Northern Raccoon, Virginia Opossum, foxes, coyotes, deer, squirrels, cats, and dogs—and returns labels that Scrypted can save on event thumbnails.

The repository includes weights for every Scrypted backend that currently implements custom classification:

| Scrypted plugin | Best fit | Config URL |
| --- | --- | --- |
| CoreML | Apple Silicon | `https://media.githubusercontent.com/media/tman9590/Yard-Critters/v1.0.3/models/coreml/config.json` |
| OpenVINO | Intel CPU/iGPU/NPU | `https://media.githubusercontent.com/media/tman9590/Yard-Critters/v1.0.3/models/openvino/config.json` |
| ONNX | NVIDIA, Windows, general CPU fallback | `https://media.githubusercontent.com/media/tman9590/Yard-Critters/v1.0.3/models/onnx/config.json` |
| NCNN | ARM/Vulkan and lightweight Linux installs | `https://media.githubusercontent.com/media/tman9590/Yard-Critters/v1.0.3/models/ncnn/config.json` |

The root [`config.json`](config.json) is the universal manifest and single reference for all backend URLs. Scrypted's loader still expects the backend-specific URL from the table because each runtime has a different native weight format. Use the version-pinned `media.githubusercontent.com` URLs exactly as shown: GitHub's `raw.githubusercontent.com` endpoint returns Git LFS pointer text instead of the model bytes, which causes `INVALID_PROTOBUF` and equivalent load errors.

## Install in Scrypted

1. Install or open the backend plugin appropriate for the Scrypted host.
2. Add a **Custom Object Detection** device.
3. Paste the matching raw `config.json` URL above into **Model URL**.
4. In each camera's Object Detector settings, select the new Yard Critters classifier.
5. Enable animal classification and recording/search metadata for the camera.

Use **one classifier backend per camera**. Running duplicate classifiers increases latency and can make metadata arrive after the thumbnail is committed.

## Why this model

SpeciesNet is a camera-trap model trained for wildlife imagery. Yard Critters uses the `always_crop` classifier because Scrypted already supplies a crop from its animal detector. This avoids feeding a tight crop to a model trained for full frames—the mismatch that commonly causes inaccurate labels.

The model accepts a 224×224 RGB crop from Scrypted, resizes it to SpeciesNet's native 480×480 resolution inside the model graph, and outputs raw logits. Keeping the external input at 224×224 avoids Scrypted's `upscale not supported` failure on small detection crops. Scrypted applies softmax and records classifications at or above its built-in confidence threshold.

## Backend support note

Scrypted's TensorFlow Lite plugin does not currently expose the custom-classifier interface, so there is no TFLite package to load. The four directories in `models/` cover all Scrypted plugins that do expose that interface as of this release.

## Rebuild and validate

The checked-in weights are Git LFS objects. To regenerate them from the official SpeciesNet release:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-export.txt
.venv/bin/python scripts/export_models.py \
  --source-model /path/to/always_crop_99710272_22x8_v12_epoch_00148.pt \
  --labels /path/to/always_crop_99710272_22x8_v12_epoch_00148.labels.20260609.txt
.venv/bin/python scripts/validate.py
```

## Accuracy and metadata

Classification cannot repair a missed animal detection; it only labels detector crops. For best results, use a detector that produces a tight `animal` box, avoid heavily compressed substreams for classification, and keep only one classifier assigned to each camera. The original SpeciesNet labels are preserved so Scrypted search metadata describes the actual species instead of forcing every animal into a short generic list.

## License and attribution

Code and converted model artifacts are provided under Apache-2.0. The model derives from [Google SpeciesNet](https://github.com/google/cameratrapai); see [`NOTICE`](NOTICE) for provenance. Model behavior and outputs remain subject to the upstream model's limitations.
