"""模型服务必须记录实际加载路径，拒绝未提交检查点。"""
import dataclasses
import json
from types import SimpleNamespace

import pytest

from tools.medication_sorting import pi05_adapter


@dataclasses.dataclass
class Config:
    policy_metadata: dict


def test_service_records_actual_loaded_committed_checkpoint(tmp_path):
    checkpoint = tmp_path / "1499"
    (checkpoint / "params").mkdir(parents=True)
    (checkpoint / "assets/local/train").mkdir(parents=True)
    (checkpoint / "_CHECKPOINT_METADATA").write_text(json.dumps({"commit_timestamp_nsecs": 123}))
    (checkpoint / "params/_METADATA").write_text("params metadata")
    (checkpoint / "params/manifest.ocdbt").write_bytes(b"manifest")
    (checkpoint / "assets/local/train/norm_stats.json").write_text("{}")
    config = Config({"robot_type": "G1"})
    loaded = []

    def load(config, path, **kwargs):
        loaded.append((config, path, kwargs))
        return SimpleNamespace(metadata=config.policy_metadata)

    module = SimpleNamespace(create_trained_policy=load)
    pi05_adapter.configure_policy_provenance(module)
    policy = module.create_trained_policy(config, checkpoint, default_prompt="sort")
    assert len(loaded) == 1
    assert loaded[0][1] == checkpoint.resolve()
    assert config.policy_metadata == {"robot_type": "G1"}
    source = policy.metadata["checkpoint_provenance"]
    assert source["path"] == str(checkpoint.resolve())
    assert source["step"] == 1499
    assert source["commit_timestamp_nsecs"] == 123
    assert len(source["params_manifest_sha256"]) == 64
    assert len(source["norm_stats_sha256"]) == 64


def test_uncommitted_checkpoint_rejected_before_model_load(tmp_path):
    checkpoint = tmp_path / "1499"
    checkpoint.mkdir()
    (checkpoint / "_CHECKPOINT_METADATA").write_text(json.dumps({"commit_timestamp_nsecs": 0}))
    module = SimpleNamespace(create_trained_policy=lambda *a, **kw: pytest.fail("loaded uncommitted model"))
    pi05_adapter.configure_policy_provenance(module)
    with pytest.raises(ValueError, match="提交"):
        module.create_trained_policy(Config({}), checkpoint)
