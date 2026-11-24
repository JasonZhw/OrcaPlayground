"""Verify source hashes, scene references and absence of private deployment data."""

import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
DIRECTORIES = ("assets", "configs", "docs", "reference", "scripts",
               "simulation", "tests", "third_party")
PRIVATE = (
    re.compile(r"(?i)[a-z]:[\\/](?:users|zxj)[\\/]"),
    re.compile(r"/root/" + r"gpufree-data/"),
    re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
)


def public_files(root=ROOT):
    files = [path for path in root.iterdir() if path.is_file()]
    for name in DIRECTORIES:
        files.extend(p for p in (root / name).rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
    return sorted(files)


def validate_files(root=ROOT):
    identities = json.loads((root / "configs/input_files.sha256.json").read_text(encoding="utf-8"))
    for name, expected in identities.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Original source identity differs: {name}")
    files = public_files(root)
    for path in files:
        if path.stat().st_size >= 100 * 1024**2:
            raise ValueError(f"Large files belong in model/dataset storage: {path.name}")
        if path.suffix == ".stl":
            continue
        text = path.read_text(encoding="utf-8")
        # Package versions with four numeric components are not network addresses.
        if path.name.startswith("requirements"):
            text = re.sub(r"(?m)^[A-Za-z0-9_.-]+==[A-Za-z0-9.+_-]+\r?$", "", text)
        # Loopback is the configurable local simulation/streaming interface.
        text = text.replace("127.0.0.1", "")
        if any(pattern.search(text) for pattern in PRIVATE):
            raise ValueError(f"Private address, path or credential pattern: {path.relative_to(root)}")
    for path in (root / "assets").glob("*.xml"):
        for element in ET.parse(path).findall("./asset/*[@file]"):
            asset = Path(element.get("file"))
            if asset.is_absolute() or not (path.parent / asset).is_file():
                raise ValueError(f"Missing or nonportable scene asset: {path.name}")
    return {"public_files": len(files), "original_hashes": len(identities),
            "portable_xml_assets": True, "privacy_patterns": "passed"}


if __name__ == "__main__":
    print(json.dumps(validate_files(), indent=2))
