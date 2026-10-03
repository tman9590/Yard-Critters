import json
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_backend_urls_use_git_lfs_media_endpoint():
    manifest = json.loads((ROOT / "config.json").read_text())
    for backend in manifest["backends"].values():
        assert backend["config_url"].startswith(
            "https://media.githubusercontent.com/media/"
        )


def test_remote_backend_files_are_content_not_lfs_pointers():
    manifest = json.loads((ROOT / "config.json").read_text())
    for backend in manifest["backends"].values():
        config_path = ROOT / backend["config"]
        config = json.loads(config_path.read_text())
        base_url = backend["config_url"].rsplit("/", 1)[0]
        for relative in config["files"]:
            url = f"{base_url}/{relative}"
            request = urllib.request.Request(url, headers={"Range": "bytes=0-127"})
            with urllib.request.urlopen(request, timeout=30) as response:
                prefix = response.read(128)
            assert not prefix.startswith(
                b"version https://git-lfs.github.com/spec/v1"
            ), url
