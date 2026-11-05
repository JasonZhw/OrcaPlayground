"""User-run upload; default is a local preflight, never a network write."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def verified_account(api):
    """Retry transient account-check transport failures, never authentication errors."""
    from requests.exceptions import ConnectionError, Timeout

    for attempt in range(3):
        try:
            return api.whoami()["name"]
        except (ConnectionError, Timeout):
            if attempt == 2:
                raise RuntimeError(
                    "HF account check could not reach the service after 3 attempts. "
                    "No upload was started by this run; check the SSH proxy connection."
                ) from None
            print("HF account-check connection interrupted; retrying shortly...", flush=True)
            time.sleep(2 * (attempt + 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Actually publish the public dataset")
    args = parser.parse_args()
    folder = args.folder.resolve()
    manifest = json.loads((folder.parent / "upload_manifest.json").read_text())
    assert manifest["repo_id"] == "Bigluck77/g1_med_sort"
    actual = {str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file()}
    assert actual == set(manifest["files"]), "Upload file set changed; regenerate/review the manifest"
    for name, expected in manifest["files"].items():
        path = folder / name
        assert not path.is_symlink()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        assert path.stat().st_size == expected["bytes"] and digest.hexdigest() == expected["sha256"], name
    print(f"Verified {len(actual)} files; public dataset Bigluck77/g1_med_sort", flush=True)
    if not args.execute:
        print("Preflight only. Nothing uploaded. Use --execute after interactive login.")
        return
    os.environ["HF_HUB_OFFLINE"] = "0"
    from huggingface_hub import HfApi

    api = HfApi()
    account = verified_account(api)
    if account != "Bigluck77":
        raise RuntimeError(f"Logged in as {account}; expected Bigluck77")
    api.create_repo("Bigluck77/g1_med_sort", repo_type="dataset", private=False, exist_ok=True)
    if api.repo_info("Bigluck77/g1_med_sort", repo_type="dataset").private:
        raise RuntimeError("Existing repository is private; set its visibility to public before this upload")
    result = api.upload_folder(repo_id="Bigluck77/g1_med_sort", repo_type="dataset",
                               folder_path=folder, allow_patterns=sorted(actual),
                               commit_message="Upload validated 20-episode G1 medication sorting dataset")
    # LeRobot v2.1 readers use this version tag. Never silently move an existing tag.
    api.create_tag("Bigluck77/g1_med_sort", repo_type="dataset", tag="v2.1",
                   revision=result.oid, exist_ok=False)
    write_result = {"commit": result.oid, "repo_id": "Bigluck77/g1_med_sort", "tag": "v2.1"}
    (folder.parent / "upload_result.json").write_text(json.dumps(write_result, indent=2) + "\n")
    print("Uploaded:", result.commit_url)


if __name__ == "__main__":
    main()
