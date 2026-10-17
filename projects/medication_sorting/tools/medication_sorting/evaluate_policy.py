"""顺序运行真实 VLA 推理回合并保存独立报告；本模块导入不启动任何服务。

仅在主线程释放采集/专家控制、完成模型服务协调后显式执行：
    /home/pc/ZXJ/SY/.venv/bin/python /home/pc/ZXJ/SY/tools/medication_sorting/evaluate_policy.py --checkpoint-reference /caller/provided/checkpoint --seed-start 413 --episodes 5 --steps 150 --execute-chunk 4 --control-hz 5 --output /absolute/new_evaluation

每次只运行一个 infer_policy 子进程并等待其退出，转发 steps/chunk/Hz、
seed、相机后端和服务地址。项目共享非阻塞锁阻止评估驱动并发；启动
每个回合前只读检查 /proc，已有本项目采集/专家/推理进程则拒绝启动。
其他入口不使用此锁，仍需由主线程协调控制权；本驱动不会停止它们。
本地 loopback 策略服务的子进程移除大小写 ALL/HTTP/HTTPS/WS/WSS_PROXY，
合并并补齐 NO_PROXY/no_proxy；只更改子环境，父进程及远程服务代理不变。
该防护不覆盖直接调用 infer_policy 的入口，后者须由主线程排除本地代理。

每个 seed 独立输出到 seed_<seed>/，原始 report.json 不改写；evaluation.json
保存驱动验收结论，启动失败/缺报告也有独立记录，stdout/stderr 保存到 logs/。
summary.json 汇总严格布尔 success/grasp_verified、
子进程退出码、缺失/损坏报告和错误。成功须退出码为零、两项验证均为
true、无报告 error，且 controller=pi05_checkpoint_policy、未用专家动作。
失败回合仍保留，任一失败返回 1；全成功返回 0。success_rate 为严格
验收成功数/已记录任务数，启动失败及缺报告均计失败。默认 5 回合、
每回合最多 150 步、5 Hz；新输出目录不覆盖。

checkpoint-reference 仅是调用方提供的说明，不加载权重，也不配置或
证明远端服务实际使用该 checkpoint。413 起的 seeds 是请求的评估种子，
尚未与训练清单核对，不能伪称已证明 heldout。训练、GPU/模型服务及
真实 rollout 验收由主线程协调；mock 测试结果不能当作 VLA 实测成功。

训练期间可纯离线核对来源（不占控制锁、不启动 child）：
    /home/pc/ZXJ/SY/.venv/bin/python /home/pc/ZXJ/SY/tools/medication_sorting/evaluate_policy.py --audit-only --training-manifest /absolute/split_manifest.json --seed-start 413 --episodes 5
记录清单哈希、train/validation seed、源 report 的 reset 和首条 trace 实测
box pose。首条 trace 是动作后的观察，并非精确的 t=0 位姿。eval reset
只按与采集/推理相同的 NumPy 算法计算预期值，不冒充实测；相同采样
范围内的新 seed 不能证明场景分布外泛化。实际评估可传同一清单，重叠
seed 启动前拒绝，并核对返回报告的 seed/reset。清单内容 hash 只检查
跨分区声明互斥，不重新证明 NPZ 内容；checkpoint 使用该训练集的来源
仍由主线程单独确认，本工具不证明远端权重来源。
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / ".venv/bin/python"
EXECUTION_LOCK = ROOT / "data/medication_sorting/policy_evaluations/.evaluation.lock"
CONTROLLERS = {"collect_episodes", "collect_balanced", "run_expert", "run_four_objects", "infer_policy", "evaluate_policy"}


def active_controllers():
    """只读取本项目控制进程；不连接、发信号或改变仿真。"""
    active = []
    for folder in Path("/proc").iterdir():
        if not folder.name.isdigit() or int(folder.name) == os.getpid():
            continue
        try:
            args = (folder / "cmdline").read_bytes().decode(errors="replace").split("\0")
            names = {Path(arg).stem for arg in args if arg.endswith(".py")}
            modules = {arg.removeprefix("tools.medication_sorting.") for arg in args
                       if arg.startswith("tools.medication_sorting.")}
            if not CONTROLLERS.intersection(names | modules):
                continue
            cwd = (folder / "cwd").resolve()
            if cwd.is_relative_to(ROOT) or any(str(ROOT) in arg for arg in args):
                active.append({"pid": int(folder.name), "command": args})
        except (OSError, RuntimeError):
            continue  # 进程在读取期间退出，或与本项目无关且无权限。
    return active


@contextmanager
def _execution_lock():
    EXECUTION_LOCK.parent.mkdir(parents=True, exist_ok=True)
    with EXECUTION_LOCK.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("已有评估驱动持有控制锁，拒绝并发评估") from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _assert_idle():
    active = active_controllers()
    if active:
        raise RuntimeError(f"本项目已有控制进程，须协调结束后再评估：{active}")


def _child_environment(host):
    """本地策略连接直连；不修改父环境，也不改变远程代理配置。"""
    environment = dict(os.environ)
    hostname = urlsplit(host if "://" in host else "//" + host).hostname
    local = hostname == "localhost"
    if not local and hostname:
        try:
            local = ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            pass
    if not local:
        return environment, "inherited_for_non_loopback"
    for key in tuple(environment):
        if key.lower() in {"all_proxy", "http_proxy", "https_proxy", "ws_proxy", "wss_proxy"}:
            environment.pop(key)
    exclusions = [item.strip() for key in ("NO_PROXY", "no_proxy")
                  for item in environment.get(key, "").split(",") if item.strip()]
    exclusions.extend(("127.0.0.1", "localhost", "::1", "[::1]", hostname))
    environment["NO_PROXY"] = environment["no_proxy"] = ",".join(dict.fromkeys(exclusions))
    return environment, "loopback_child_proxy_excluded"


def _report(path):
    def reject_constant(value):
        raise ValueError(f"JSON 不允许非有限值：{value}")
    def finite_float(value):
        number = float(value)
        if not math.isfinite(number):
            reject_constant(value)
        return number
    try:
        report = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant, parse_float=finite_float)
    except (OSError, ValueError) as error:
        return None, f"独立推理 report 缺失或损坏：{error}"
    if not isinstance(report, dict):
        return None, "独立推理 report 必须为 JSON 对象"
    return report, None


def _expected_reset(seed, *, profile='legacy'):
    # 延迟导入；离线审计无需导入 infer_policy、机器人或模型代码。
    from tools.medication_sorting.randomization import sample_reset
    offset, yaw = sample_reset(seed, profile=profile)
    return {"seed": seed, "offset": offset.tolist(), "yaw": yaw, "observed": False}


def _reset_matches(report, expected):
    import numpy as np
    try:
        offset = np.asarray(report.get("reset_offset"), dtype=float)
        yaw = report.get("reset_yaw")
        return bool(offset.shape == (2,) and np.isfinite(offset).all()
                    and np.allclose(offset, expected["offset"], atol=1e-12, rtol=0)
                    and type(yaw) in (int, float) and math.isfinite(yaw)
                    and abs(yaw - expected["yaw"]) <= 1e-12)
    except (TypeError, ValueError):
        return False


def audit_heldout_seeds(manifest_path, seeds, *, randomization_profile='legacy'):
    """只读核对已落盘 seed/reset 与首条真实 trace，不启动仿真。"""
    import numpy as np
    manifest_path = Path(manifest_path).resolve()
    seeds = list(seeds)
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("评估 seed 必须是非负、不重复的整数")
    manifest_bytes = manifest_path.read_bytes()
    manifest, error = _report(manifest_path)
    if error or manifest.get("status") != "materialized":
        raise ValueError(error or "必须提供已实际建立划分的 manifest")
    records, saved_seeds, hashes = [], {}, {}
    for partition in ("train", "validation"):
        entries = manifest.get(partition)
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"manifest.{partition} 必须有独立回合")
        saved_seeds[partition], hashes[partition] = [], set()
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("source"), str):
                raise ValueError("manifest 回合必须指向实际 source")
            source = Path(entry["source"]).resolve()
            if entry.get("link") and Path(entry["link"]).resolve() != source:
                raise ValueError(f"manifest link 与 source 不一致：{source}")
            digest = entry.get("content_sha256")
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("manifest content hash 必须为 SHA256")
            hashes[partition].add(digest)
            metadata, error = _report(source / "metadata.json")
            if error or metadata.get("success") is not True or metadata.get("complete") is not True:
                raise ValueError(error or f"source metadata 未提交成功：{source}")
            seed = metadata.get("seed")
            if type(seed) is not int or seed < 0 or metadata.get("randomized_reset") is not True:
                raise ValueError(f"source 必须有真实 randomized seed：{source}")
            report, error = _report(source / "report.json")
            if error or report.get("success") is not True:
                raise ValueError(error or f"source report 未成功：{source}")
            source_profile = metadata.get('randomization_profile', 'legacy')
            if not _reset_matches(report, _expected_reset(seed, profile=source_profile)):
                raise ValueError(f"source reset 与保存 seed 不符：{source}")
            trace_path = source / "trace.jsonl"
            with trace_path.open(encoding="utf-8") as trace:
                first_line = trace.readline()
            observation = json.loads(first_line)
            if (not isinstance(observation, dict) or observation.get("phase") != "settle_initial"
                    or type(observation.get("time")) not in (int, float)
                    or not math.isfinite(observation["time"]) or observation["time"] <= 0):
                raise ValueError(f"source 首条 trace 缺少初始阶段实测：{source}")
            for key, length in (("box_pos", 3), ("box_quat", 4)):
                vector = np.asarray(observation.get(key), dtype=float)
                if vector.shape != (length,) or not np.isfinite(vector).all():
                    raise ValueError(f"source 首条 trace.{key} 无效：{source}")
            saved_seeds[partition].append(seed)
            records.append({"partition": partition, "source": str(source), "seed": seed,
                            "content_sha256_from_manifest": digest, "reset_matches_seed": True,
                            "reset_offset": report["reset_offset"], "reset_yaw": report["reset_yaw"],
                            "first_trace_observation": {key: observation[key] for key in ("time", "phase", "box_pos", "box_quat")},
                            "evidence_paths": {"metadata": str(source / "metadata.json"), "report": str(source / "report.json"), "trace": str(trace_path)}})
    if hashes["train"] & hashes["validation"]:
        raise ValueError("manifest train/validation content hash 泄漏")
    if manifest_path.read_bytes() != manifest_bytes:
        raise ValueError("审计期间 manifest 变化，请重新审计")
    overlaps = {partition: sorted(set(seeds) & set(saved_seeds[partition])) for partition in saved_seeds}
    return {"manifest": str(manifest_path), "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "train_seeds": sorted(set(saved_seeds["train"])), "validation_seeds": sorted(set(saved_seeds["validation"])),
            "requested_seeds": seeds, "seed_disjoint_from_train": not overlaps["train"],
            "seed_disjoint_from_validation": not overlaps["validation"],
            "overlap_train_seeds": overlaps["train"], "overlap_validation_seeds": overlaps["validation"],
            "source_episodes": records, "evaluation_expected_resets": [_expected_reset(seed, profile=randomization_profile) for seed in seeds],
            "evaluation_initial_poses_observed": False, "checkpoint_training_provenance_verified": False,
            "manifest_content_hashes_reverified": False, "read_only": True,
            "evidence_limit": "仅证明与给定划分的 seed 分离及来源 reset 一致；首条 trace 为动作后观察，eval 位姿未实测，远端 checkpoint 训练来源未核实。"}


def evaluate_policy(output, *, checkpoint_reference, episodes=5, seed_start=413,
                    max_steps=150, execute_chunk=4, control_hz=5,
                    host="127.0.0.1", port=8000, camera_backend="viewport", training_manifest=None,
                    target=None, profile='original', layout=None, randomization_profile='legacy'):
    """显式执行入口；单回合失败不删除证据，继续下一个独立 seed。"""
    if (any(type(value) is not int or value <= 0 for value in (episodes, max_steps, execute_chunk))
            or type(seed_start) is not int or seed_start < 0
            or type(control_hz) is not int or control_hz not in (5, 10, 20)
            or type(port) is not int or not 1 <= port <= 65535
            or camera_backend not in ("viewport", "sensor")):
        raise ValueError("episodes/steps/chunk 必须为正整数，seed 非负，control-hz 为 5/10/20，端口及相机配置有效")
    if not isinstance(checkpoint_reference, str) or not checkpoint_reference.strip():
        raise ValueError("必须明确给出调用方的 checkpoint-reference")
    from tools.medication_sorting.randomization import profile_config
    from tools.medication_sorting.task_specs import target_spec
    profile_config(randomization_profile)
    if target is not None:
        target_spec(target, profile=profile)
    if Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
        raise RuntimeError(f"唯一允许的 Python 环境为 {ROOT / '.venv'}")
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"输出已存在，拒绝覆盖：{output}")
    child_environment, proxy_policy = _child_environment(host)
    result = {"checkpoint_reference": {"value": checkpoint_reference, "source": "caller_supplied", "verified": False},
              "heldout_status": "not_verified_against_training_manifest",
              "created_at": datetime.now(timezone.utc).isoformat(), "output": str(output),
              "controls": {"steps": max_steps, "execute_chunk": execute_chunk, "control_hz": control_hz,
                           "camera_backend": camera_backend, "host": host, "port": port, "proxy_policy": proxy_policy},
              "requested_seeds": list(range(seed_start, seed_start + episodes)), "episodes": [],
              "task_count": 0, "success_count": 0, "success_rate": 0.0, "failed_count": 0, "grasp_verified_count": 0,
              "error_count": 0, "exit_code": 1, "sequential": True}
    if training_manifest is not None:
        audit = audit_heldout_seeds(training_manifest, result["requested_seeds"], randomization_profile=randomization_profile)
        if not audit["seed_disjoint_from_train"] or not audit["seed_disjoint_from_validation"]:
            raise ValueError("评估 seed 与给定训练/验证清单重叠，拒绝启动")
        result.update(heldout_audit=audit, heldout_status="seed_disjoint_from_supplied_manifest_only")
    with _execution_lock():
        _assert_idle()
        output.mkdir(parents=True)
        logs = output / "logs"
        logs.mkdir()
        for seed in result["requested_seeds"]:
            folder = output / f"seed_{seed}"
            command = [str(PYTHON), "-m", "tools.medication_sorting.infer_policy", "--host", host, "--port", str(port),
                       "--steps", str(max_steps), "--execute-chunk", str(execute_chunk), "--control-hz", str(control_hz),
                       "--camera-backend", camera_backend, "--seed", str(seed), "--output", str(folder)]
            if target is not None:
                command += ['--target', target, '--profile', profile,
                            '--randomization-profile', randomization_profile]
                if layout is not None:
                    command += ['--layout', str(layout)]
            record = {"seed": seed, "output": str(folder), "report_path": str(folder / "report.json"),
                      "success": False, "grasp_verified": False, "error": None, "returncode": None,
                      "command": command}
            stdout, stderr = "", ""
            try:
                _assert_idle()
                completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False, shell=False,
                                           env=child_environment)
                record["returncode"] = completed.returncode
                stdout, stderr = completed.stdout or "", completed.stderr or ""
                report, error = _report(folder / "report.json")
                if report is not None:
                    record["report_success"] = report.get("success") is True
                    record["grasp_verified"] = report.get("grasp_verified") is True
                    record["error"] = report.get("error") or error
                    provenance_ok = report.get("controller") == "pi05_checkpoint_policy" and report.get("expert_actions_used") is False
                    if target is not None:
                        spec = target_spec(target, profile=profile)
                        record['target_matches'] = (report.get('object_name') == target
                            and report.get('task') == spec['task']
                            and report.get('expert_profile') == profile
                            and report.get('randomization_profile') == randomization_profile)
                        metadata = report.get('policy_metadata')
                        provenance = metadata.get('checkpoint_provenance') if isinstance(metadata, dict) else None
                        provenance = provenance if isinstance(provenance, dict) else {}
                        record['checkpoint_path_matches'] = provenance.get('path') == str(Path(checkpoint_reference).resolve())
                        provenance_ok = provenance_ok and record['target_matches'] and record['checkpoint_path_matches']
                    if training_manifest is not None:
                        record["reported_reset_matches_seed"] = report.get("seed") == seed and _reset_matches(report, _expected_reset(seed, profile=randomization_profile))
                        provenance_ok = provenance_ok and record["reported_reset_matches_seed"]
                    record["success"] = bool(completed.returncode == 0 and record["report_success"]
                                              and record["grasp_verified"] and not record["error"] and provenance_ok)
                    if not record["success"] and not record["error"]:
                        record["error"] = "推理退出码、success/grasp_verified 或策略控制声明未满足验收"
                else:
                    record["error"] = error
            except (OSError, RuntimeError) as error:
                record["error"] = f"{type(error).__name__}: {error}"
            (logs / f"seed_{seed}.stdout.txt").write_text(stdout, encoding="utf-8")
            (logs / f"seed_{seed}.stderr.txt").write_text(stderr, encoding="utf-8")
            # infer_policy 自己独占创建输出目录；仅在其退出后补独立驱动记录。
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "evaluation.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            result["episodes"].append(record)
            result.update(task_count=len(result["episodes"]),
                          success_count=sum(row["success"] for row in result["episodes"]),
                          failed_count=sum(not row["success"] for row in result["episodes"]),
                          grasp_verified_count=sum(row["grasp_verified"] for row in result["episodes"]),
                          error_count=sum(bool(row["error"]) for row in result["episodes"]))
            result["success_rate"] = result["success_count"] / result["task_count"]
            result["exit_code"] = int(bool(result["failed_count"]))
            (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="顺序评估真实 infer_policy 回合；不训练、不加载模型，checkpoint 引用仅由调用方提供。")
    parser.add_argument("--checkpoint-reference")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--training-manifest", type=Path, help="可选：核对实际划分清单的 seed/reset；不证明 checkpoint 训练来源")
    parser.add_argument("--audit-only", action="store_true", help="只读来源审计，不启动推理或占用控制锁")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=413)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--execute-chunk", type=int, default=4)
    parser.add_argument("--control-hz", type=int, choices=(5, 10, 20), default=5)
    parser.add_argument("--camera-backend", choices=("viewport", "sensor"), default="viewport")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument('--target')
    parser.add_argument('--profile', default='original')
    parser.add_argument('--layout', type=Path)
    parser.add_argument('--randomization-profile', default='legacy')
    args = parser.parse_args(argv)
    if args.audit_only:
        if args.training_manifest is None:
            parser.error("--audit-only 要求 --training-manifest")
        try:
            if args.episodes <= 0 or args.seed_start < 0:
                raise ValueError("episodes 必须为正，seed 必须非负")
            result = audit_heldout_seeds(args.training_manifest, range(args.seed_start, args.seed_start + args.episodes),
                                         randomization_profile=args.randomization_profile)
            result["exit_code"] = int(not result["seed_disjoint_from_train"] or not result["seed_disjoint_from_validation"])
        except (ValueError, OSError) as error:
            result = {"status": "blocked", "read_only": True, "error": str(error), "exit_code": 1}
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return result["exit_code"]
    if args.output is None or not args.checkpoint_reference:
        parser.error("执行评估要求 --output 和 --checkpoint-reference")
    try:
        result = evaluate_policy(args.output, checkpoint_reference=args.checkpoint_reference, episodes=args.episodes,
                                 seed_start=args.seed_start, max_steps=args.steps, execute_chunk=args.execute_chunk,
                                 control_hz=args.control_hz, camera_backend=args.camera_backend, host=args.host, port=args.port,
                                 training_manifest=args.training_manifest, target=args.target, profile=args.profile,
                                 layout=args.layout, randomization_profile=args.randomization_profile)
    except (ValueError, OSError, RuntimeError) as error:
        result = {"status": "blocked", "error": str(error), "exit_code": 1}
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return result["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
