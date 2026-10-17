"""G1 外置适配器的数值、采集质量及锁定接口测试；全部在 CPU 上运行。"""

import ast
import abc
import dataclasses
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from enum import Enum
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
OPENPI = ROOT / "software/openpi"
JOINTS = [f"right_joint_{i}" for i in range(7)]


def test_adapter_implementation_required_for_numeric_contract():
    assert importlib.util.find_spec("tools.medication_sorting.pi05_adapter") is not None, "缺少数值契约实现"


def test_converter_implementation_required_for_data_contract():
    assert importlib.util.find_spec("tools.medication_sorting.convert_episodes") is not None, "缺少完整回合质量门禁实现"


@pytest.fixture(scope="module")
def adapter():
    name = "tools.medication_sorting.pi05_adapter"
    assert importlib.util.find_spec(name) is not None, "缺少 G1 外置适配器实现"
    return importlib.import_module(name)


@pytest.fixture(scope="module")
def converter():
    name = "tools.medication_sorting.convert_episodes"
    assert importlib.util.find_spec(name) is not None, "缺少成功回合转换器实现"
    return importlib.import_module(name)


@pytest.fixture
def observation():
    return {
        "state": np.array([0.4, -0.3, 1.1, -1.2, 0.2, -0.1, 0.8, 0.6], dtype=np.float32),
        "base_rgb": np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3),
        "wrist_rgb": np.arange(60, 120, dtype=np.uint8).reshape(4, 5, 3),
        "prompt": "Put the medicine box into the purple basket.",
    }


@pytest.fixture
def raw_episode(tmp_path):
    directory = tmp_path / "raw" / "episode_001"
    directory.mkdir(parents=True)
    arrays = {
        "state": np.zeros((4, 8), dtype=np.float32),
        "action": np.full((4, 8), 0.2, dtype=np.float32),
        "base_rgb": np.tile(np.arange(60, dtype=np.uint8).reshape(1, 4, 5, 3), (4, 1, 1, 1)),
        "wrist_rgb": np.tile(np.arange(60, 120, dtype=np.uint8).reshape(1, 4, 5, 3), (4, 1, 1, 1)),
        "sim_time": np.array([12.0, 12.05, 12.10, 12.15], dtype=np.float64),
        "phase": np.array(["approach", "grasp", "place", "release"]),
    }
    metadata = {
        "success": True,
        "task": "Put the medicine box into the purple basket.",
        "joint_names": JOINTS.copy(),
        "gripper_semantics": {"open": 0.0, "closed": 1.0},
        "camera_metadata": {
            "base_rgb": {"name": "base", "color_space": "RGB", "layout": "HWC"},
            "wrist_rgb": {"name": "right_wrist", "color_space": "RGB", "layout": "HWC"},
        },
    }

    def write():
        np.savez(directory / "episode.npz", **arrays)
        (directory / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    write()
    return directory, arrays, metadata, write


@pytest.fixture
def official_transforms(monkeypatch):
    """直接执行 pinned OpenPI 的 NumPy 变换源码，省去 JAX/Flax/tokenizer 导入。

    不重写 delta、padding 或 quantile 数学。树工具仅替代未安装的 jax.tree.map。
    """
    names = {
        "Group", "RepackTransform", "DeltaActions", "AbsoluteActions", "PadStatesAndActions",
        "pad_to_dim", "make_bool_mask", "Normalize", "Unnormalize",
        "InjectDefaultPrompt", "ResizeImages", "TokenizePrompt",
    }
    source = OPENPI / "src/openpi/transforms.py"
    tree = ast.parse(source.read_text())
    selected = [node for node in tree.body if getattr(node, "name", None) in names]
    assert len(selected) == len(names)
    module = ModuleType("openpi.transforms")

    def tree_map(fn, value):
        if isinstance(value, dict):
            return {key: tree_map(fn, item) for key, item in value.items()}
        return fn(value)

    def flatten(value, prefix=""):
        result = {}
        for key, item in value.items():
            name = f"{prefix}/{key}" if prefix else key
            result.update(flatten(item, name) if isinstance(item, dict) else {name: item})
        return result

    module.__dict__.update(
        np=np, dataclasses=dataclasses, DataTransformFn=object,
        jax=SimpleNamespace(tree=SimpleNamespace(map=tree_map)), flatten_dict=flatten,
    )
    monkeypatch.setitem(sys.modules, "openpi", ModuleType("openpi"))
    monkeypatch.setitem(sys.modules, "openpi.transforms", module)
    code = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected], type_ignores=[])
    exec(compile(ast.fix_missing_locations(code), str(source), "exec"), module.__dict__)
    return module


@pytest.fixture
def official_config(monkeypatch, official_transforms):
    """执行 pinned 配置类源码；替换未安装的框架、下载器和 tokenizer 边界。

    配置字段、create_base_config、ModelTransformFactory 和 LoRA filter 方法
    来自实际上游源码；此测试不声称运行了真实 tokenizer 或训练栈。
    """
    class ModelType(Enum):
        PI0 = "pi0"
        PI05 = "pi05"
        PI0_FAST = "pi0_fast"

    nnx = SimpleNamespace(Nothing="nothing", Not=lambda x: ("not", x), All=lambda *x: ("all", *x))
    model_base = SimpleNamespace(ModelType=ModelType, BaseModelConfig=object)
    model_module = ModuleType("openpi.models.pi0_config")
    model_module.__dict__.update(dataclasses=dataclasses, _model=model_base, nnx=nnx,
                                 nnx_utils=SimpleNamespace(PathRegex=lambda x: x), override=lambda x: x)
    config_module = ModuleType("openpi.training.config")
    paths = []

    def no_stats(path):
        paths.append(path)
        raise FileNotFoundError(path)

    import logging
    weights = ModuleType("openpi.training.weight_loaders")
    weights.NoOpWeightLoader = lambda: None
    weights.CheckpointWeightLoader = lambda path: SimpleNamespace(params_path=path)
    config_module.__dict__.update(
        abc=abc, dataclasses=dataclasses, pathlib=importlib.import_module("pathlib"), logging=logging,
        tyro=SimpleNamespace(MISSING=object()), nnx=nnx, _model=model_base, ModelType=ModelType,
        GroupFactory=object, override=lambda x: x, _transforms=official_transforms,
        _tokenizer=SimpleNamespace(PaligemmaTokenizer=lambda size: SimpleNamespace(max_token_len=size)),
        _normalize=SimpleNamespace(load=no_stats), _download=SimpleNamespace(maybe_download=lambda x: x),
        epath=SimpleNamespace(Path=Path), weight_loaders=weights,
        _optimizer=SimpleNamespace(CosineDecaySchedule=lambda: None, AdamW=lambda: None),
        pi0_config=model_module,
    )
    for name, obj in (("openpi.models", ModuleType("openpi.models")),
                      ("openpi.training", ModuleType("openpi.training")),
                      (model_module.__name__, model_module), (config_module.__name__, config_module),
                      (weights.__name__, weights)):
        monkeypatch.setitem(sys.modules, name, obj)

    def execute_classes(path, names, module):
        source = ast.parse(path.read_text())
        nodes = [node for node in source.body if getattr(node, "name", None) in names]
        assert len(nodes) == len(names)
        code = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
        exec(compile(ast.fix_missing_locations(code), str(path), "exec"), module.__dict__)

    execute_classes(OPENPI / "src/openpi/models/pi0_config.py", {"Pi0Config"}, model_module)
    execute_classes(OPENPI / "src/openpi/training/config.py",
                    {"AssetsConfig", "DataConfig", "ModelTransformFactory", "DataConfigFactory", "FakeDataConfig", "SimpleDataConfig", "TrainConfig"}, config_module)
    config_module._CONFIGS_DICT = {"pi05_droid": object()}
    return config_module, paths


def test_right_wrist_slot_and_missing_left_mask(adapter, observation):
    result = adapter.G1MedicationInputs()(observation)
    np.testing.assert_array_equal(result["image"]["base_0_rgb"], observation["base_rgb"])
    np.testing.assert_array_equal(result["image"]["right_wrist_0_rgb"], observation["wrist_rgb"])
    assert not result["image"]["left_wrist_0_rgb"].any()
    assert result["image_mask"] == {"base_0_rgb": True, "left_wrist_0_rgb": False, "right_wrist_0_rgb": True}
    assert result["state"].shape == (8,)
    assert result["state"].dtype == np.float32
    assert result["prompt"] == observation["prompt"]


def test_lerobot_float32_chw_rgb_roundtrip(adapter, observation):
    for key in ("base_rgb", "wrist_rgb"):
        image = observation[key].copy()
        observation[key] = image.transpose(2, 0, 1).astype(np.float32) / 255.0
        result = adapter.G1MedicationInputs()(observation)
        slot = "base_0_rgb" if key == "base_rgb" else "right_wrist_0_rgb"
        np.testing.assert_array_equal(result["image"][slot], image)


@pytest.mark.parametrize("kind", ["dtype", "width", "nan", "closure"])
def test_invalid_observation_state_rejected(adapter, observation, kind):
    if kind == "dtype":
        observation["state"] = observation["state"].astype(np.float64)
    elif kind == "width":
        observation["state"] = observation["state"][:7]
    elif kind == "nan":
        observation["state"][2] = np.nan
    else:
        observation["state"][-1] = 1.01
    with pytest.raises(ValueError, match="state"):
        adapter.G1MedicationInputs()(observation)


@pytest.mark.parametrize("kind", ["range", "nan", "channels", "shape"])
def test_invalid_lerobot_images_rejected(adapter, observation, kind):
    image = observation["base_rgb"].transpose(2, 0, 1).astype(np.float32) / 255
    if kind == "range":
        image[0, 0, 0] = 1.01
    elif kind == "nan":
        image[0, 0, 0] = np.nan
    elif kind == "channels":
        image = image[:1]
    else:
        image = image[0]
    observation["base_rgb"] = image
    with pytest.raises(ValueError, match="base_rgb"):
        adapter.G1MedicationInputs()(observation)


def test_official_delta_seven_joints_padding_and_inverse(adapter, observation, official_transforms):
    absolute = np.array([
        [0.9, -0.5, 1.4, -1.0, 0.5, 0.1, 0.7, 0.9],
        [1.0, -0.4, 1.5, -0.9, 0.6, 0.2, 0.6, 0.1],
    ], dtype=np.float32)
    observation["actions"] = absolute.copy()
    original_state = observation["state"].copy()
    group = adapter.make_data_transforms()
    data = observation
    for transform in group.inputs:
        data = transform(data)
    expected = absolute.copy()
    expected[:, :7] -= original_state[:7]
    np.testing.assert_allclose(data["actions"], expected, atol=1e-7)
    np.testing.assert_array_equal(observation["actions"], absolute)
    np.testing.assert_array_equal(data["actions"][:, 7], absolute[:, 7])
    padded = official_transforms.PadStatesAndActions(32)(data)
    assert padded["state"].shape == (32,)
    assert padded["actions"].shape == (2, 32)
    assert not padded["actions"][:, 8:].any()
    assert not padded["state"][8:].any()
    padded["actions"][:, 8:] = 123.0  # 输出丢弃 padding，不能累加到真实关节。
    for transform in group.outputs:
        padded = transform(padded)
    assert padded["actions"].dtype == np.float32
    np.testing.assert_allclose(padded["actions"], absolute, atol=2e-7)


def test_quantile_unnormalize_precedes_absolute_joint_restore(adapter, observation, official_transforms):
    group = adapter.make_data_transforms()
    observation["actions"] = np.tile(observation["state"], (3, 1)) + np.float32(0.1)
    expected = observation["actions"].copy()
    data = observation
    for transform in group.inputs:
        data = transform(data)
    state_stats = SimpleNamespace(q01=np.full(8, -2, np.float32), q99=np.full(8, 2, np.float32))
    action_stats = SimpleNamespace(q01=np.full(8, -0.5, np.float32), q99=np.full(8, 1.5, np.float32))
    normalizer = official_transforms.Normalize(None, use_quantiles=True)
    inverse = official_transforms.Unnormalize(None, use_quantiles=True)
    for key, stats in (("state", state_stats), ("actions", action_stats)):
        data[key] = normalizer._normalize_quantile(data[key], stats)
    data = official_transforms.PadStatesAndActions(32)(data)
    for key, stats in (("state", state_stats), ("actions", action_stats)):
        data[key] = inverse._unnormalize_quantile(data[key], stats)
    for transform in group.outputs:
        data = transform(data)
    np.testing.assert_allclose(data["actions"], expected, atol=4e-7)


def test_model_gripper_output_clipped_without_clipping_joints(adapter):
    model_actions = np.zeros((2, 32), dtype=np.float64)
    model_actions[:, 0] = [-3.2, 4.3]
    model_actions[:, 7] = [-0.05, 1.05]
    result = adapter.G1MedicationOutputs()({"actions": model_actions})["actions"]
    assert result.shape == (2, 8)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result[:, 0], [-3.2, 4.3])
    np.testing.assert_array_equal(result[:, 7], [0, 1])
    np.testing.assert_array_equal(model_actions[:, 7], [-0.05, 1.05])


def test_nonfinite_model_output_rejected(adapter):
    actions = np.zeros((1, 32), dtype=np.float32)
    actions[0, 2] = np.inf
    with pytest.raises(ValueError, match="actions"):
        adapter.G1MedicationOutputs()({"actions": actions})


def test_schema_uses_official_chw_features_and_absolute_raw_actions(converter, raw_episode):
    directory, arrays, _, _ = raw_episode
    episode = converter.load_episode(directory, fps=20)
    features = converter.make_features(episode)
    assert features["observation.images.base"]["shape"] == (3, 4, 5)
    assert features["observation.images.right_wrist"]["shape"] == (3, 4, 5)
    assert features["observation.state"]["names"] == JOINTS + ["gripper_closure"]
    frames = list(converter.iter_frames(episode))
    assert len(frames) == 4
    np.testing.assert_array_equal(frames[0]["action"], arrays["action"][0])
    assert frames[0]["observation.images.base"].dtype == np.uint8
    assert frames[0]["task"] == episode.metadata["task"]
    assert "timestamp" not in frames[0]  # 官方根据 fps 生成；绝不把绝对仿真时刻当 timestamp。
    assert frames[0]["sim_time"].shape == (1,)
    assert frames[0]["phase"] == "approach"


@pytest.mark.parametrize("kind", [
    "missing_action", "length", "state_dtype", "action_nan", "closure", "empty_image",
    "image_dtype", "image_shape", "duplicate_views", "time_nan", "time_reverse",
    "time_gap", "empty_phase", "object_phase", "joint_jump",
])
def test_corrupt_episode_rejected(converter, raw_episode, kind):
    directory, arrays, _, write = raw_episode
    if kind == "missing_action":
        del arrays["action"]
    elif kind == "length":
        arrays["wrist_rgb"] = arrays["wrist_rgb"][:-1]
    elif kind == "state_dtype":
        arrays["state"] = arrays["state"].astype(np.float64)
    elif kind == "action_nan":
        arrays["action"][0, 2] = np.nan
    elif kind == "closure":
        arrays["action"][0, 7] = -0.1
    elif kind == "empty_image":
        arrays["wrist_rgb"][2] = 0
    elif kind == "image_dtype":
        arrays["base_rgb"] = arrays["base_rgb"].astype(np.float32)
    elif kind == "image_shape":
        arrays["wrist_rgb"] = arrays["wrist_rgb"][:, :3]
    elif kind == "duplicate_views":
        arrays["wrist_rgb"] = arrays["base_rgb"].copy()
    elif kind == "time_nan":
        arrays["sim_time"][1] = np.nan
    elif kind == "time_reverse":
        arrays["sim_time"][2] = arrays["sim_time"][1]
    elif kind == "time_gap":
        arrays["sim_time"][2:] += 0.03
    elif kind == "empty_phase":
        arrays["phase"][2] = ""
    elif kind == "object_phase":
        arrays["phase"] = arrays["phase"].astype(object)
    else:
        arrays["action"][2, 0] = 2.0
    write()
    with pytest.raises(ValueError):
        converter.load_episode(directory, fps=20)


@pytest.mark.parametrize("kind", ["failed", "success_string", "incomplete", "task", "joint_names", "gripper", "bgr", "camera_missing"])
def test_invalid_metadata_rejected(converter, raw_episode, kind):
    directory, _, metadata, write = raw_episode
    if kind == "failed":
        metadata["success"] = False
    elif kind == "success_string":
        metadata["success"] = "true"
    elif kind == "incomplete":
        metadata["complete"] = False
    elif kind == "task":
        metadata["task"] = "把药盒放入紫色筐"
    elif kind == "joint_names":
        metadata["joint_names"][1] = metadata["joint_names"][0]
    elif kind == "gripper":
        metadata["gripper_semantics"] = {"open": 1, "closed": 0}
    elif kind == "bgr":
        metadata["camera_metadata"]["base_rgb"]["color_space"] = "BGR"
    else:
        del metadata["camera_metadata"]["wrist_rgb"]
    write()
    with pytest.raises(ValueError):
        converter.load_episode(directory, fps=20)


def test_terminal_phase_and_joint_order_guard(converter, raw_episode):
    directory, _, _, _ = raw_episode
    with pytest.raises(ValueError, match="phase"):
        converter.load_episode(directory, fps=20, terminal_phase="verified")
    with pytest.raises(ValueError, match="joint_names"):
        converter.load_episode(directory, fps=20, expected_joint_names=tuple(reversed(JOINTS)))


class RecordingLeRobotDataset:
    """签名来自锁定 LeRobotDataset.create/add_frame/save_episode，无 consolidate/finalize。"""

    created = []

    @classmethod
    def create(cls, repo_id, fps, root=None, robot=None, robot_type=None, features=None, use_videos=True,
               tolerance_s=1e-4, image_writer_processes=0, image_writer_threads=0, video_backend=None):
        obj = cls()
        obj.kwargs = {"repo_id": repo_id, "fps": fps, "root": root, "robot_type": robot_type,
                      "features": features, "use_videos": use_videos}
        obj.frames = []
        obj.episodes = []
        cls.created.append(obj)
        Path(root).mkdir(parents=True, exist_ok=False)
        return obj

    def add_frame(self, frame):
        assert set(frame) == set(self.kwargs["features"]) | {"task"}
        self.frames.append(frame)

    def save_episode(self, episode_data=None):
        assert episode_data is None
        self.episodes.append(self.frames)
        self.frames = []

    def stop_image_writer(self):
        pass


def test_conversion_filters_failed_and_incomplete_before_writer(converter, raw_episode, tmp_path):
    directory, arrays, metadata, _ = raw_episode
    bad = directory.parent / "episode_002"
    bad.mkdir()
    metadata = {**metadata, "success": False}
    (bad / "metadata.json").write_text(json.dumps(metadata))
    np.savez(bad / "episode.npz", **arrays)
    output = tmp_path / "converted"
    report = converter.convert_episodes(directory.parent, output, repo_id="local/g1_medication", fps=20,
                                        dataset_class=RecordingLeRobotDataset)
    assert report["converted_episodes"] == 1
    assert report["converted_frames"] == 4
    assert len(report["rejected"]) == 1
    writer = RecordingLeRobotDataset.created[-1]
    assert writer.kwargs["robot_type"] == "unitree_g1_fixed_base_right_arm"
    assert writer.kwargs["use_videos"] is False
    assert len(writer.episodes) == 1
    manifest = json.loads((output / "g1_conversion_manifest.json").read_text())
    assert manifest["joint_names"] == JOINTS
    assert manifest["openpi_commit"] == "215abfb217dbac7d5f1273282331b9b1866c0479"


def test_joint_order_change_is_rejected_before_conversion(converter, raw_episode, tmp_path):
    directory, arrays, metadata, _ = raw_episode
    second = directory.parent / "episode_002"
    second.mkdir()
    other = {**metadata, "joint_names": list(reversed(JOINTS))}
    (second / "metadata.json").write_text(json.dumps(other))
    np.savez(second / "episode.npz", **arrays)
    report = converter.convert_episodes(directory.parent, tmp_path / "out", repo_id="local/g1_medication", fps=20,
                                        dataset_class=RecordingLeRobotDataset)
    assert report["converted_episodes"] == 1
    assert "joint_names" in report["rejected"][0]["reason"]


def test_no_good_episodes_never_creates_output(converter, raw_episode, tmp_path):
    directory, _, metadata, write = raw_episode
    metadata["complete"] = False
    write()
    before = len(RecordingLeRobotDataset.created)
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="成功"):
        converter.convert_episodes(directory.parent, output, repo_id="local/g1_medication", fps=20,
                                   dataset_class=RecordingLeRobotDataset)
    assert len(RecordingLeRobotDataset.created) == before
    assert not output.exists()


def test_existing_output_preserved(converter, raw_episode, tmp_path):
    directory, _, _, _ = raw_episode
    output = tmp_path / "out"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("原数据")
    with pytest.raises(FileExistsError):
        converter.convert_episodes(directory.parent, output, repo_id="local/g1_medication", fps=20,
                                   dataset_class=RecordingLeRobotDataset)
    assert sentinel.read_text() == "原数据"


@pytest.mark.parametrize("repo_id", ["droid", "../g1", "local/../../g1", "/tmp/g1", "local/.."])
def test_repo_id_cannot_escape_assets_or_cache(adapter, repo_id):
    with pytest.raises(ValueError, match="repo_id"):
        adapter.validate_repo_id(repo_id)


def test_pinned_checkout_and_lightweight_import(adapter):
    adapter.check_openpi_checkout(OPENPI)
    script = "import sys; import tools.medication_sorting.pi05_adapter; import tools.medication_sorting.convert_episodes; assert not {'jax','torch','lerobot','openpi'} & sys.modules.keys()"
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_missing_lerobot_reports_pinned_revision_without_install(converter, monkeypatch):
    def missing(_):
        raise importlib.metadata.PackageNotFoundError("lerobot")
    monkeypatch.setattr(importlib.metadata, "distribution", missing)
    with pytest.raises(RuntimeError, match="0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"):
        converter.require_lerobot()


def test_lerobot_same_version_wrong_commit_rejected(converter, monkeypatch):
    package = SimpleNamespace(read_text=lambda _: json.dumps({"vcs_info": {"commit_id": "wrong"}}))
    monkeypatch.setattr(importlib.metadata, "distribution", lambda _: package)
    with pytest.raises(RuntimeError, match="commit"):
        converter.require_lerobot()


def test_gripper_metadata_boolean_is_not_numeric_semantics(converter, raw_episode):
    directory, _, metadata, write = raw_episode
    metadata["gripper_semantics"] = {"open": False, "closed": True}
    write()
    with pytest.raises(ValueError, match="gripper_semantics"):
        converter.load_episode(directory, fps=20)


def test_npy_disguised_as_npz_isolated_by_scan(converter, raw_episode):
    directory, _, _, _ = raw_episode
    with (directory / "episode.npz").open("wb") as stream:
        np.save(stream, np.zeros((4, 8), dtype=np.float32))
    accepted, rejected = converter.scan_episodes(directory.parent, fps=20)
    assert accepted == []
    assert len(rejected) == 1
    assert "npz" in rejected[0]["reason"]


def test_invalid_fps_rejected_even_for_empty_input(converter, tmp_path):
    with pytest.raises(ValueError, match="fps"):
        converter.scan_episodes(tmp_path, fps=0)


def test_data_config_rejects_invalid_repo_before_dependency_import(adapter, monkeypatch):
    monkeypatch.setitem(sys.modules, "openpi", None)
    with pytest.raises(ValueError, match="repo_id"):
        adapter.make_data_config("../g1")


def test_gripper_delta_regression_is_numerically_detected(adapter, observation, official_transforms):
    """防止把第八维一起减 state；非零初始 closure 才能发现这个错误。"""
    observation["state"][7] = np.float32(0.75)
    observation["actions"] = np.array([[0.4, -0.3, 1.1, -1.2, 0.2, -0.1, 0.8, 0.25]], dtype=np.float32)
    group = adapter.make_data_transforms()
    for transform in group.inputs:
        observation = transform(observation)
    assert observation["actions"][0, 7] == np.float32(0.25)
    np.testing.assert_array_equal(observation["actions"][0, :7], np.zeros(7, np.float32))


def test_official_factory_repack_quantile_own_assets_and_model_padding(adapter, observation, official_config, tmp_path):
    config_module, paths = official_config
    factory = adapter.make_data_config("local/g1_medication")
    model = config_module.pi0_config.Pi0Config(pi05=True)
    data_config = factory.create(tmp_path, model)
    assert data_config.asset_id == "local/g1_medication"
    assert paths == [str(tmp_path / "local/g1_medication")]
    assert data_config.norm_stats is None
    assert data_config.use_quantile_norm is True
    assert data_config.prompt_from_task is True
    assert data_config.action_sequence_keys == ("action",)
    assert data_config.rlds_data_dir is None
    raw = {"observation.state": observation["state"], "observation.images.base": observation["base_rgb"],
           "observation.images.right_wrist": observation["wrist_rgb"],
           "action": np.tile(observation["state"], (2, 1)), "prompt": observation["prompt"]}
    repacked = data_config.repack_transforms.inputs[0](raw)
    assert set(repacked) == {"state", "base_rgb", "wrist_rgb", "actions", "prompt"}
    assert type(data_config.model_transforms.inputs[-1]).__name__ == "PadStatesAndActions"
    assert data_config.model_transforms.inputs[-1].model_action_dim == 32
    assert data_config.model_transforms.inputs[-2].discrete_state_input is True
    assert [type(x).__name__ for x in data_config.data_transforms.outputs] == ["AbsoluteActions", "G1MedicationOutputs"]


def test_official_train_config_lora_and_external_registry(adapter, official_config):
    config_module, _ = official_config
    train_config = adapter.create_train_config("local/g1_medication", action_horizon=12, batch_size=1)
    assert train_config.model.pi05 is True
    assert train_config.model.action_dim == 32
    assert train_config.model.action_horizon == 12
    assert train_config.model.paligemma_variant == "gemma_2b_lora"
    assert train_config.model.action_expert_variant == "gemma_300m_lora"
    assert train_config.freeze_filter == train_config.model.get_freeze_filter()
    assert train_config.ema_decay is None
    assert train_config.pytorch_weight_path is None
    assert train_config.weight_loader.params_path == "gs://openpi-assets/checkpoints/pi05_base/params"
    previous = config_module._CONFIGS_DICT["pi05_droid"]
    adapter.register_config(train_config)
    assert config_module._CONFIGS_DICT[train_config.name] is train_config
    assert config_module._CONFIGS_DICT["pi05_droid"] is previous
    with pytest.raises(ValueError, match="已注册"):
        adapter.register_config(train_config)


@pytest.mark.parametrize("global_arguments", [
    ["--port=8001", "--default-prompt=Put the box in the basket."],
    ["--port", "8001", "--default-prompt", "Put the box in the basket.", "--record"],
    ["--help"],
])
def test_serve_root_options_precede_checkpoint_subcommand(adapter, monkeypatch, global_arguments):
    monkeypatch.setattr(adapter, "check_openpi_checkout", lambda: None)
    monkeypatch.setattr(adapter, "configure_policy_provenance", lambda: None)
    monkeypatch.setattr(adapter, "create_train_config", lambda *args, **kwargs: object())
    monkeypatch.setattr(adapter, "register_config", lambda _: None)
    calls = []
    monkeypatch.setattr(adapter.runpy, "run_path", lambda *args, **kwargs: calls.append(adapter.sys.argv.copy()))
    adapter.run_openpi_entrypoint('serve', repo_id='local/g1_medication',
                                 arguments=['--policy.dir=/tmp/checkpoint/0', *global_arguments])
    command = calls[0]
    subcommand = command.index('policy:checkpoint')
    assert command[1:subcommand] == global_arguments
    assert command[subcommand + 1:] == ['--policy.config', adapter.CONFIG_NAME, '--policy.dir=/tmp/checkpoint/0']
