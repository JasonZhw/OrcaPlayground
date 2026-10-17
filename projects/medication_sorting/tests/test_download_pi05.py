"""离线复现官方组合对象的校验漏洞；不启动真实下载或 GPU。"""

import base64
import hashlib
import io
import json
from pathlib import Path

import pytest

from tools.medication_sorting import download_pi05 as download


def item_for(data=b"123456789", *, md5=True, crc=True):
    item = {"name": download.PREFIX + "params/example", "size": str(len(data))}
    if md5:
        item["md5Hash"] = base64.b64encode(hashlib.md5(data).digest()).decode()
    if crc:
        # CRC32C/Castagnoli 标准向量；不是 zlib 的 IEEE CRC32。
        assert data == b"123456789"
        item["crc32c"] = "4waSgw=="
    return item


def test_composite_same_size_corruption_is_replaced_after_checksum_check(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "DEST", tmp_path)
    item = item_for(md5=False)
    path = tmp_path / "params/example"
    path.parent.mkdir()
    path.write_bytes(b"987654321")
    requests = []

    class OfflineOpener:
        def open(self, request, timeout):
            requests.append(request)
            response = io.BytesIO(b"123456789")
            response.status = 200
            return response

    monkeypatch.setattr(download.urllib.request, "build_opener", lambda *args: OfflineOpener())
    assert download.fetch(item) == str(path)
    assert path.read_bytes() == b"123456789", "同长度损坏组合对象不能仅凭长度被接受"
    assert len(requests) == 1


def test_object_without_any_official_checksum_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "DEST", tmp_path)
    path = tmp_path / "params/example"
    path.parent.mkdir()
    path.write_bytes(b"123456789")
    with pytest.raises(ValueError, match="校验"):
        download.fetch(item_for(md5=False, crc=False))


def test_verify_only_checks_entire_manifest_without_network(tmp_path, monkeypatch):
    item = item_for(md5=False)
    path = tmp_path / "params/example"
    path.parent.mkdir()
    path.write_bytes(b"123456789")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"items": [item]}))

    def forbidden(*args, **kwargs):
        pytest.fail("离线复验不得启动下载")

    monkeypatch.setattr(download.urllib.request, "build_opener", forbidden)
    report = download.verify_manifest(manifest, tmp_path)
    assert report["all_files_verified"] is True
    assert report["md5_verified"] is False
    assert report["crc32c_verified"] is True
    assert report["verified_files"] == report["total_files"] == 1
    assert report["checks"][0]["actual_crc32c"] == item["crc32c"]


@pytest.mark.parametrize("contents", [None, b"987654321", b"123"])
def test_missing_corrupt_or_truncated_file_fails_manifest_verification(tmp_path, contents):
    path = tmp_path / "params/example"
    path.parent.mkdir()
    if contents is not None:
        path.write_bytes(contents)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"items": [item_for(crc=False)]}))
    report = download.verify_manifest(manifest, tmp_path)
    assert report["all_files_verified"] is False
    assert report["verified_files"] == 0
    assert report["checks"][0]["verified"] is False


def test_empty_checkpoint_marker_has_real_checksum(tmp_path):
    path = tmp_path / "params/example"
    path.parent.mkdir()
    path.touch()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"items": [item_for(b"", crc=False)]}))
    report = download.verify_manifest(manifest, tmp_path)
    assert report["all_files_verified"] is True
    assert report["md5_verified"] is True
    assert report["checks"][0]["actual_size"] == 0
