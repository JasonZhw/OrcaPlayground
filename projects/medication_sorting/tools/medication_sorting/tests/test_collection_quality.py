"""数据质量离线测试；所有人工样例只放 pytest 临时目录，不能作为实测成功证据。"""

import importlib
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import pytest

from tools.medication_sorting.convert_episodes import load_episode, scan_episodes


JOINTS = ["unitree_g1_1_right_" + name + "_joint" for name in (
    "shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
PHASES = ["settle_initial", "approach", "descend", "close", "lift", "carry", "lower", "release", "retreat"]


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def rewrite_npz(path, **changes):
    with np.load(path / "episode.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays.update(changes)
    np.savez_compressed(path / "episode.npz", **arrays)


@pytest.fixture
def episode(tmp_path):
    """完整格式的人工样例，用来证明校验能拒绝各类坏数据。"""
    path = tmp_path / "raw" / "episode_0000"
    path.mkdir(parents=True)
    n = 30
    phase = np.array(PHASES + ["settle_final"] * (n - len(PHASES)))
    state = np.zeros((n, 8), dtype=np.float32)
    state[:, 0] = np.arange(n) * 0.005
    state[:, 7] = 0.5
    action = state.copy()
    action[:, 0] += 0.005
    action[:, 7] = [0.5, 0.5, 0.7, 1, 1, 0.8, 0.6, 0.3, 0] + [0] * (n - 9)
    state[1:, 7] = action[:-1, 7]
    times = np.arange(n, dtype=np.float64) / 5
    images = {}
    camera_meta = {}
    for key, name, seed in [("base_rgb", "head_cam", 2), ("wrist_rgb", "camera_right", 3)]:
        raw = np.random.default_rng(seed).integers(0, 256, (n, 8, 8, 3), dtype=np.uint8)
        images[key] = np.rint(255 * (raw.astype(np.float32) / 255) ** 0.45).astype(np.uint8)
        camera_meta[key] = {"name": name, "color_space": "RGB", "layout": "HWC", "dtype": "uint8",
                            "renderer": "OrcaLab native", "gamma": 0.45, "resolution": [8, 8]}
        for i, image in enumerate(raw):
            folder = path / "frames" / f"{i:06d}" / "color"
            folder.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image).save(folder / f"{name}_color_0.png")
    np.savez_compressed(path / "episode.npz", state=state, action=action,
                        sim_time=times, phase=phase, **images)
    metadata = {"success": True, "complete": True, "synchronized": True, "fps": 5, "frames": n,
                "task": "Put the medicine box into the purple basket.", "joint_names": JOINTS,
                "gripper_semantics": {"open": 0, "closed": 1}, "camera_metadata": camera_meta,
                "camera_alignment": "paused_physics_snapshot_with_camera_pose_check",
                "test_only_synthetic_fixture": True}
    write_json(path / "metadata.json", metadata)
    trace = []
    for i in range(n):
        lifted = phase[i] in {"lift", "carry", "lower"}
        trace.append({"time": float(times[i] + 0.2), "phase": str(phase[i]), "action": action[i].tolist(),
                      "state": state[min(i + 1, n - 1)].tolist(), "box_pos": [0, 0, 0.25 if lifted else 0],
                      "contact_sides": ["inner", "outer"] if phase[i] in {"close", "lift", "carry", "lower"} else []})
    (path / "trace.jsonl").write_text("".join(json.dumps(row) + "\n" for row in trace), encoding="utf-8")
    report = {"success": True, "lift_verified": True, "lift_height": 0.25, "released": True,
              "placement": {"inside": True, "local_bounds": [[-0.1, -0.1, 0], [0.1, 0.1, 0.1]],
                            "floor_half_size": [0.2, 0.2], "floor_height": 0},
              "frames": n, "final_linear_speed": 0.001,
              "no_object_teleport_during_episode": True}
    write_json(path / "report.json", report)
    return path


def verify(path):
    return importlib.import_module("tools.medication_sorting.verify_episodes").verify_episode(path)


def test_offline_verifier_accepts_consistent_fixture_and_measures_rgb(episode):
    result = verify(episode)
    assert result["accepted"], result["errors"]
    assert result["stats"]["frames"] == 30
    assert result["stats"]["base_rgb"]["unique_frames"] == 30
    assert result["stats"]["wrist_rgb"]["adjacent_duplicate_fraction"] == 0
    assert result["stats"]["native_png_pairs_checked"] == 30
    assert result["stats"]["fps"] == 5


def test_verifier_accepts_older_20hz_only_when_times_match_metadata(episode):
    meta = json.loads((episode / "metadata.json").read_text())
    meta["fps"] = 20
    write_json(episode / "metadata.json", meta)
    with np.load(episode / "episode.npz") as archive:
        state, action = archive["state"].copy(), archive["action"].copy()
    action[:, 7] = np.r_[np.full(9, 0.04), np.zeros(21)]
    state[:, 7] = np.r_[0.04, action[:-1, 7]]
    rewrite_npz(episode, sim_time=np.arange(30, dtype=np.float64) / 20, state=state, action=action)
    trace_path = episode / "trace.jsonl"
    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    for index, row in enumerate(rows):
        row["time"] = (index + 1) / 20
        row["action"] = action[index].tolist()
        row["state"] = state[min(index + 1, 29)].tolist()
    trace_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = verify(episode)
    assert result["accepted"], result["errors"]
    assert result["stats"]["fps"] == 20


@pytest.mark.parametrize("field,value", [("fps", 10), ("frames", 999), ("complete", False),
                                        ("synchronized", False), ("success", False)])
def test_verifier_rejects_inconsistent_metadata(episode, field, value):
    meta = json.loads((episode / "metadata.json").read_text())
    meta[field] = value
    write_json(episode / "metadata.json", meta)
    assert not verify(episode)["accepted"]


@pytest.mark.parametrize("field,value", [("success", False), ("lift_verified", False), ("released", False),
                                        ("lift_height", 0.02), ("final_linear_speed", 0.1)])
def test_verifier_requires_real_report_evidence(episode, field, value):
    report = json.loads((episode / "report.json").read_text())
    report[field] = value
    write_json(episode / "report.json", report)
    assert not verify(episode)["accepted"]


def test_multi_object_metadata_cannot_label_purple_box_as_blue_bottle(episode):
    meta=json.loads((episode/'metadata.json').read_text())
    meta.update(object_name='melatonin_tablet_bottle_1',basket_suffix='_Box_Blue')
    write_json(episode/'metadata.json',meta)
    report=json.loads((episode/'report.json').read_text())
    report.update(object_name='pharyngitis_tablets_box_1',
                  task='Put the medicine box into the purple basket.',basket='test_Box_Purple')
    write_json(episode/'report.json',report)
    assert not verify(episode)['accepted'], '四物品数据不能混用单药盒的标签或紫筐证据'


@pytest.mark.parametrize('metadata_profile,report_profile', [
    ('scaled_bottles_70', 'original'), ('original', 'scaled_bottles_70'),
    ('unknown', 'unknown'), ('scaled_bottles_70', None),
])
def test_verifier_rejects_inconsistent_experiment_profile(episode, metadata_profile, report_profile):
    meta=json.loads((episode/'metadata.json').read_text())
    report=json.loads((episode/'report.json').read_text())
    meta['expert_profile']=metadata_profile
    if report_profile is not None:
        report['expert_profile']=report_profile
    write_json(episode/'metadata.json',meta)
    write_json(episode/'report.json',report)
    result=verify(episode)
    assert not result['accepted'], '缩小实验不能与原尺寸配置混标'
    assert any('配置' in message for message in result['errors'])


@pytest.mark.parametrize('profile', ['original', 'scaled_bottles_70'])
def test_verifier_accepts_matching_experiment_profile(episode, profile):
    for name in ['metadata.json', 'report.json']:
        content=json.loads((episode/name).read_text())
        content['expert_profile']=profile
        write_json(episode/name,content)
    assert verify(episode)['accepted']


@pytest.mark.parametrize('name', ['calcium_gluconate_tablets_bottle_1','melatonin_tablet_bottle_1',
                                'fu_shi_gao_herb_drink_packaging_box_1','pharyngitis_tablets_box_1'])
def test_multi_object_labels_must_match_canonical_task_and_actual_body(episode,name):
    from tools.medication_sorting.run_expert import TARGETS
    spec=TARGETS[name]
    meta=json.loads((episode/'metadata.json').read_text())
    report=json.loads((episode/'report.json').read_text())
    meta.update(object_name=name,basket_suffix=spec['basket_suffix'],task=spec['task'])
    report.update(object_name=name,task=spec['task'],box='test'+spec['body_suffix'],
                  basket='test'+spec['basket_suffix'])
    write_json(episode/'metadata.json',meta)
    write_json(episode/'report.json',report)
    assert verify(episode)['accepted']
    report['box']='test_Medical_999'
    write_json(episode/'report.json',report)
    assert not verify(episode)['accepted'], '相互一致的标签仍必须与实际物品刚体绑定'
    report['box']='test'+spec['body_suffix']
    report['task']=meta['task']='Put an unrelated object somewhere else.'
    write_json(episode/'metadata.json',meta)
    write_json(episode/'report.json',report)
    assert not verify(episode)['accepted'], '两份JSON共同错误的任务不能通过验收'


def test_verifier_rejects_bounds_outside_recorded_floor_even_if_inside_is_true(episode):
    report=json.loads((episode/'report.json').read_text())
    report['placement']['floor_center']=[.5,0,0]
    write_json(episode/'report.json',report)
    result=verify(episode)
    assert not result['accepted'], '报告的inside声明不能掩盖实际位于筐底范围外的几何'


def test_verifier_rejects_frozen_nonblank_camera(episode):
    with np.load(episode / "episode.npz") as archive:
        wrist = np.repeat(archive["wrist_rgb"][:1], 30, axis=0)
    rewrite_npz(episode, wrist_rgb=wrist)
    result = verify(episode)
    assert not result["accepted"]
    assert result["stats"]["wrist_rgb"]["unique_frames"] == 1


def test_alpha_channel_cannot_hide_empty_native_rgb(episode):
    png = episode / "frames/000000/color/camera_right_color_0.png"
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    rgba[:, :, 3] = 255
    Image.fromarray(rgba).save(png)
    assert not verify(episode)["accepted"]


def test_trace_state_after_action_must_match_next_observation(episode):
    path = episode / "trace.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["state"][0] += 0.2
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert not verify(episode)["accepted"]


def test_native_png_must_match_saved_exposed_rgb(episode):
    png = episode / "frames/000001/color/head_cam_color_0.png"
    Image.fromarray(np.random.default_rng(99).integers(0, 256, (8, 8, 3), dtype=np.uint8)).save(png)
    assert not verify(episode)["accepted"]


def viewport_fixture(path):
    meta = json.loads((path / "metadata.json").read_text())
    meta["camera_alignment"] = "paused_physics_snapshot_with_explicit_camera_transform"
    for key, name in [("base_rgb", "virtual_overhead"), ("wrist_rgb", "virtual_right_wrist")]:
        meta["camera_metadata"][key]["name"] = name
    write_json(path / "metadata.json", meta)
    with np.load(path / "episode.npz") as archive:
        times = archive["sim_time"]
        for index in range(30):
            folder = path / "frames" / f"{index:06d}"
            for key, name in [("base_rgb", "head_cam"), ("wrist_rgb", "camera_right")]:
                shutil.copyfile(folder / "color" / f"{name}_color_0.png", folder / f"{key}.png")
                Image.fromarray(archive[key][index]).save(folder / f"{key}_224.png")
            write_json(folder / "camera_transforms.json", {
                "frame": index, "sim_time": float(times[index]), "physics_paused": True,
                "poses": {"base_rgb": {"position": [0.58, -3.62, 5.35], "rotation_degrees": [-90, 0, 180]},
                          "wrist_rgb": {"position": [index * 0.005, -3.5, 5.1], "rotation_degrees": [90, 0, -90]}}})


def test_viewport_png_and_paused_transforms_are_checked_at_5hz(episode):
    viewport_fixture(episode)
    result = verify(episode)
    assert result["accepted"], result["errors"]
    assert result["stats"]["camera_backend"] == "viewport"
    assert result["stats"]["camera_transforms_checked"] == 30


@pytest.mark.parametrize("field,value", [("sim_time", 123), ("physics_paused", False), ("frame", 3)])
def test_viewport_rejects_stale_or_unpaused_transform_evidence(episode, field, value):
    viewport_fixture(episode)
    path = episode / "frames/000000/camera_transforms.json"
    transforms = json.loads(path.read_text())
    transforms[field] = value
    write_json(path, transforms)
    assert not verify(episode)["accepted"]


def test_trace_contact_cannot_be_hidden_by_report_released_flag(episode):
    path = episode / "trace.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-1]["contact_sides"] = ["inner"]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert not verify(episode)["accepted"]


def test_deployment_command_rate_is_not_hidden_by_half_rad_quality_threshold(episode):
    with np.load(episode / "episode.npz") as archive:
        action = archive["action"].copy()
    action[5, 0] += 0.3  # 小于 0.5 rad，但在 5 Hz 下超过 1.2 rad/s。
    rewrite_npz(episode, action=action)
    path = episode / "trace.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for index, row in enumerate(rows):
        row["action"] = action[index].tolist()
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = verify(episode)
    assert not result["accepted"]
    assert any("1.2 rad/s" in error for error in result["errors"])


def test_viewport_resizes_rgba_before_gamma_and_checks_transform_filename_alias(episode):
    viewport_fixture(episode)
    folder = episode / "frames/000000"
    (folder / "camera_transforms.json").rename(folder / "cameraTransforms.json")
    transforms = json.loads((folder / "cameraTransforms.json").read_text())
    transforms["simTime"] = transforms.pop("sim_time")
    write_json(folder / "cameraTransforms.json", transforms)
    with Image.open(folder / "base_rgb.png") as image:
        image = image.convert("RGBA").resize((108, 80), Image.Resampling.LANCZOS)
        image.save(folder / "base_rgb.png")
        rgb = np.asarray(image.convert("RGB").resize((8, 8), Image.Resampling.LANCZOS))
    exposed = np.rint(255 * (rgb.astype(np.float32) / 255) ** 0.45).astype(np.uint8)
    Image.fromarray(exposed).save(folder / "base_rgb_224.png")
    with np.load(episode / "episode.npz") as archive:
        base = archive["base_rgb"].copy()
    base[0] = exposed
    rewrite_npz(episode, base_rgb=base)
    result = verify(episode)
    assert result["accepted"], result["errors"]


@pytest.mark.parametrize("source_key", ["ee_position", "actual_sourceEE_position"])
@pytest.mark.parametrize("wrong_source", [False, True])
def test_hand_follower_uses_synchronous_source_ee_for_all_frames(episode, wrong_source, source_key):
    viewport_fixture(episode)
    metadata = json.loads((episode / "metadata.json").read_text())
    metadata["camera_metadata"]["wrist_rgb"]["name"] = "virtual_hand_follower"
    write_json(episode / "metadata.json", metadata)
    for index in range(30):
        path = episode / "frames" / f"{index:06d}" / "camera_transforms.json"
        transforms = json.loads(path.read_text())
        pose = transforms["poses"]["wrist_rgb"]
        pose["rotation_degrees"] = [-65, 0, 180]
        source = np.asarray(pose["position"]) - [-0.035, 0.16, 0.28]
        if wrong_source and index == 0:
            source[0] += 0.1
        transforms[source_key] = source.tolist()
        write_json(path, transforms)
    result = verify(episode)
    assert result["accepted"] is not wrong_source, result["errors"]
    if not wrong_source:
        assert result["stats"]["hand_follower_direct_ee_checks"] == 30


def test_viewport_audits_retained_blank_retry_image(episode):
    viewport_fixture(episode)
    for index in range(30):
        path = episode / "frames" / f"{index:06d}" / "camera_transforms.json"
        transforms = json.loads(path.read_text())
        transforms["render_attempts"] = {"base_rgb": 1, "wrist_rgb": 2 if index == 0 else 1}
        write_json(path, transforms)
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(episode / "frames/000000/rejected_wrist_rgb_01.png")
    result = verify(episode)
    assert result["accepted"], result["errors"]
    assert result["stats"]["render_retries"] == 1
    assert result["stats"]["retained_rejected_pngs_checked"] == 1
    (episode / "frames/000000/rejected_wrist_rgb_01.png").unlink()
    assert not verify(episode)["accepted"]


def test_converter_allows_legacy_report_absence_but_strict_verifier_rejects(episode):
    (episode / "report.json").unlink()
    assert load_episode(episode, fps=5).length == 30
    assert not verify(episode)["accepted"]


def test_strict_verifier_requires_saved_basket_floor_geometry(episode):
    path = episode / "report.json"
    report = json.loads(path.read_text())
    report["placement"] = {"inside": True}
    write_json(path, report)
    assert not verify(episode)["accepted"]


def test_converter_rejects_single_color_frames_even_when_color_changes_over_time(episode):
    rgb = np.zeros((30, 8, 8, 3), dtype=np.uint8)
    rgb[:, :, :, 0] = np.arange(30)[:, None, None] + 100
    rewrite_npz(episode, wrist_rgb=rgb)
    with pytest.raises(ValueError, match="空白/单色"):
        load_episode(episode, fps=5)


def test_converter_rejects_command_rate_violation_below_jump_threshold(episode):
    with np.load(episode / "episode.npz") as archive:
        action = archive["action"].copy()
    action[5, 0] += 0.3
    rewrite_npz(episode, action=action)
    with pytest.raises(ValueError, match="1.2 rad/s"):
        load_episode(episode, fps=5)


def test_duplicate_episodes_cannot_create_a_validation_split(episode):
    shutil.copytree(episode, episode.parent / "episode_0001")
    result = importlib.import_module("tools.medication_sorting.verify_episodes").verify_dataset(episode.parent)
    assert len(result["accepted"]) == 2
    assert result["split"]["status"] == "blocked"
    assert result["split"]["distinct_content_groups"] == 1
    assert not result["ok"]


def test_suggested_split_uses_whole_distinct_episodes_and_is_not_claimed_applied(episode):
    second = episode.parent / "episode_0001"
    shutil.copytree(episode, second)
    png = second / "frames/000000/color/head_cam_color_0.png"
    with Image.open(png) as image:
        raw = np.array(image.convert("RGB"))
    raw[0, 0] = 255 - raw[0, 0]
    Image.fromarray(raw).save(png)
    with np.load(second / "episode.npz") as archive:
        base = archive["base_rgb"].copy()
    base[0] = np.rint(255 * (raw.astype(np.float32) / 255) ** 0.45).astype(np.uint8)
    rewrite_npz(second, base_rgb=base)
    module = importlib.import_module("tools.medication_sorting.verify_episodes")
    result = module.verify_dataset(episode.parent)
    assert result["ok"], result
    assert result["split"]["status"] == "suggested_only"
    assert result["split"]["applied_to_training_loader"] is False
    assert len(result["split"]["train"]) == len(result["split"]["validation"]) == 1
    assert set(result["split"]["train"]).isdisjoint(result["split"]["validation"])
    assert result["split"] == module.verify_dataset(episode.parent)["split"]


@pytest.mark.parametrize("times", [np.arange(30) * 0.05, np.r_[0, 0, np.arange(2, 30) * 0.2],
                                  np.arange(30) * 0.2 + np.r_[np.zeros(15), np.full(15, 0.2)]])
def test_converter_rejects_wrong_frequency_duplicate_time_and_dropped_frame(episode, times):
    rewrite_npz(episode, sim_time=times)
    with pytest.raises(ValueError, match="sim_time"):
        load_episode(episode, fps=5)


def test_converter_excludes_failed_episode_even_when_npz_is_complete(episode):
    meta = json.loads((episode / "metadata.json").read_text())
    meta["success"] = False
    write_json(episode / "metadata.json", meta)
    accepted, rejected = scan_episodes(episode.parent, fps=5)
    assert accepted == []
    assert len(rejected) == 1


def test_converter_must_not_accept_report_failure_hidden_by_metadata(episode):
    report = json.loads((episode / "report.json").read_text())
    report.update(success=False, released=False)
    write_json(episode / "report.json", report)
    accepted, _ = scan_episodes(episode.parent, fps=5)
    assert accepted == [], "metadata:true 不能掩盖 report 中的失败"


def test_converter_must_reject_entire_frozen_camera_stream(episode):
    with np.load(episode / "episode.npz") as archive:
        wrist = np.repeat(archive["wrist_rgb"][:1], 30, axis=0)
    rewrite_npz(episode, wrist_rgb=wrist)
    accepted, _ = scan_episodes(episode.parent, fps=5)
    assert accepted == [], "非空白图片重复整段不能作为有效相机采集"


def test_converter_must_check_metadata_fps_against_requested_rate(episode):
    meta = json.loads((episode / "metadata.json").read_text())
    meta["fps"] = 10
    write_json(episode / "metadata.json", meta)
    accepted, _ = scan_episodes(episode.parent, fps=5)
    assert accepted == [], "不能忽略 metadata 中与 NPZ 不一致的 fps"


def split_module():
    return importlib.import_module("tools.medication_sorting.split_episodes")


def test_converter_and_verifier_scan_immediate_symlink_episode_children(episode, tmp_path):
    source = tmp_path / "linked_train"
    source.mkdir()
    (source / "episode_0000").symlink_to(episode, target_is_directory=True)
    accepted, rejected = scan_episodes(source, fps=5, terminal_phase="settle_final")
    assert len(accepted) == 1 and not rejected
    result = importlib.import_module("tools.medication_sorting.verify_episodes").verify_dataset(source)
    assert len(result["accepted"]) == 1


def synthetic_corpus(episode, count):
    """仅在 pytest 临时目录制作不同的人工图像内容，不是真实采集产物。"""
    for index in range(1, count):
        path = episode.parent / f"episode_{index:04d}"
        shutil.copytree(episode, path)
        png = path / "frames/000000/color/head_cam_color_0.png"
        with Image.open(png) as image:
            raw = np.array(image.convert("RGB"))
        raw[0, 0] = [index * 9 % 256, 17, 99]
        Image.fromarray(raw).save(png)
        with np.load(path / "episode.npz") as archive:
            base = archive["base_rgb"].copy()
        base[0] = np.rint(255 * (raw.astype(np.float32) / 255) ** 0.45).astype(np.uint8)
        rewrite_npz(path, base_rgb=base)


def test_split_requires_twenty_distinct_train_groups_and_independent_validation(episode, tmp_path):
    output = tmp_path / "split"
    with pytest.raises(ValueError, match="20.*1"):
        split_module().materialize_split(episode.parent, output)
    assert not output.exists()
    synthetic_corpus(episode, 20)
    with pytest.raises(ValueError, match="20.*1"):
        split_module().materialize_split(episode.parent, output)
    assert not output.exists()


def test_split_materializes_only_accepted_whole_groups_without_leakage(episode, tmp_path):
    synthetic_corpus(episode, 23)
    duplicate = episode.parent / "episode_0023"
    shutil.copytree(episode, duplicate)
    failed = episode.parent / "episode_failed"
    shutil.copytree(episode, failed)
    metadata = json.loads((failed / "metadata.json").read_text())
    metadata.update(success=False, complete=False)
    write_json(failed / "metadata.json", metadata)
    original = (episode / "episode.npz").read_bytes()
    output = tmp_path / "split"
    report = split_module().materialize_split(episode.parent, output)
    assert report["status"] == "materialized"
    assert report["train_unique_groups"] >= 20
    assert report["validation_unique_groups"] >= 1
    groups = {partition: {item["content_sha256"] for item in report[partition]} for partition in ("train", "validation")}
    assert groups["train"].isdisjoint(groups["validation"])
    assert len(report["train"]) + len(report["validation"]) == 24
    assert (output / "split_manifest.json").exists()
    assert any(item["path"] == str(failed) for item in report["excluded"])
    for partition in ("train", "validation"):
        for item in report[partition]:
            link = Path(item["link"])
            assert link.is_symlink()
            assert link.resolve() == Path(item["source"])
        accepted, rejected = scan_episodes(output / partition, fps=5, terminal_phase="settle_final")
        assert len(accepted) == len(report[partition]) and not rejected
    assert (episode / "episode.npz").read_bytes() == original
    assert report["training_loader_split_applied"] is False
    assert report["conversion"]["train_repo_id"] != report["conversion"]["validation_repo_id"]
    with pytest.raises(FileExistsError):
        split_module().materialize_split(episode.parent, output)


def test_split_refuses_an_output_inside_the_source_tree(episode):
    with pytest.raises(ValueError, match="输入目录"):
        split_module().materialize_split(episode.parent, episode.parent / "split")


def test_completed_only_quality_summary_separates_pending_and_retained_retry_pngs(episode):
    viewport_fixture(episode)
    pending = episode.parent / "episode_0001"
    pending.mkdir()
    write_json(pending / "report.json", {"success": False, "phases": []})
    folder = episode / "frames/000000"
    transforms = json.loads((folder / "camera_transforms.json").read_text())
    transforms["render_attempts"] = {"base_rgb": 2, "wrist_rgb": 1}
    write_json(folder / "camera_transforms.json", transforms)
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(folder / "rejected_base_rgb_01.png")
    module = importlib.import_module("tools.medication_sorting.verify_episodes")
    result = module.verify_dataset(episode.parent, completed_only=True)
    assert len(result["accepted"]) == 1
    assert not result["rejected"]
    assert len(result["pending"]) == 1
    summary = module.summarize_dataset(result)
    assert summary["accepted_count"] == 1 and summary["rejected_count"] == 0 and summary["pending_count"] == 1
    assert summary["render_retries"] == summary["retained_rejected_pngs_checked"] == 1
    assert summary["episodes"][0]["rgb"]["base_rgb"]["unique_frames"] == 30
    assert summary["quality_ok"] is True
    assert summary["actual_split_ready"] is False


def test_completed_only_cli_counts_frames_only_pending_and_keeps_completed_rejections(episode, capsys):
    pending = episode.parent / "episode_0001"
    (pending / "frames").mkdir(parents=True)
    failed = episode.parent / "episode_0002"
    failed.mkdir()
    write_json(failed / "metadata.json", {"success": False, "complete": False, "fps": 5})
    module = importlib.import_module("tools.medication_sorting.verify_episodes")
    assert module.main(["--input", str(episode.parent), "--completed-only", "--summary"]) == 1
    summary = json.loads(capsys.readouterr().out)
    assert summary["accepted_count"] == summary["rejected_count"] == summary["pending_count"] == 1
    assert summary["quality_ok"] is False
    assert summary["episodes"][1]["errors"]


def test_other_episode_collection_update_does_not_invalidate_completed_evidence(episode, monkeypatch):
    module = importlib.import_module("tools.medication_sorting.verify_episodes")
    write_json(episode.parent / "collection.json", [{"episode": 0, "success": True}])
    camera_checks = module._camera_checks
    def check(*args, **kwargs):
        camera_checks(*args, **kwargs)
        write_json(episode.parent / "collection.json", [{"episode": 0, "success": True}, {"episode": 1, "success": True}])
    monkeypatch.setattr(module, "_camera_checks", check)
    result = module.verify_episode(episode)
    assert result["accepted"], result["errors"]
