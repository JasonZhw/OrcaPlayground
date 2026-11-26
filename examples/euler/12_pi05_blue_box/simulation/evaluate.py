"""Record policy-only closed-loop trials in the V9 OrcaStudio scene."""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluation_metadata import provenance
from policy_runtime import PolicyRuntime
from camera_capture import Cameras
from environment import DualArmEnv
from placement import Placement, stable_frame
from task_config import ROOT, TARGET, ROBOT
from build_model import OUTPUT


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


class EvaluationEnv(DualArmEnv):
    def __init__(self):
        self.placement = None
        super().__init__()
        self.placement = Placement(self)

    def do_simulation(self, ctrl, frames):
        super().do_simulation(ctrl, frames)
        if self.placement is None:
            return
        placement = self.placement
        for contact in self.query_contact_simple():
            pair = (contact["geom1"], contact["geom2"])
            bodies = [placement.geom_bodies[g] for g in pair]
            robot = [body.startswith(ROBOT + "_") for body in bodies]
            if any(robot) and not all(robot):
                other = bodies[1] if robot[0] else bodies[0]
                touching = bodies[0] if robot[0] else bodies[1]
                if other != TARGET or "gripper_r_" not in touching:
                    raise RuntimeError("collision_abort: robot contacted environment")
            if placement.target_geom in pair:
                other_geom = pair[1] if pair[0] == placement.target_geom else pair[0]
                if (placement.geom_bodies[other_geom] == placement.basket
                        and other_geom != placement.floor_geom):
                    raise RuntimeError("collision_abort: object contacted basket wall")


def trial(env, cameras, runtime, folder, seed, max_steps, max_joint_jump):
    folder.mkdir()
    env.restore()
    runtime.reset(seed)
    command = env.setpoint.copy()
    initial_z = float(env.observe()["target"][2])
    max_lift, stable, closed_once, active_release = 0.0, 0, False, False
    previous = command[env.action_ids].copy()
    rows, chunks = [], []
    result = {"seed": seed, "success": False, "reason": "timeout"}
    log = (folder / "video.log").open("wb")
    encoder = subprocess.Popen([
        "ffmpeg", "-nostdin", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1280x480", "-r", "50",
        "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "18", "-pix_fmt", "yuv420p", str(folder / "cameras.mp4"),
    ], stdin=subprocess.PIPE, stderr=log)
    try:
        for index in range(max_steps):
            images = cameras.capture()
            if any(image.std() < 2 for image in images.values()):
                raise RuntimeError("camera_abort: blank image")
            if index == 0:
                for name, image in images.items():
                    Image.fromarray(image).save(folder / f"initial_{name}.png")
            state = env.data.qpos[env.qids[env.action_ids]].astype(np.float32)
            if index % 50 == 0:
                chunk, _ = runtime.predict(state, images["head_cam"], images["camera_right"])
                chunks.append(chunk.copy())
            raw_action = chunk[index % 50].copy()
            if raw_action.shape != (9,) or not np.isfinite(raw_action).all():
                raise RuntimeError("invalid_policy_action_abort")
            limits = env.ctrl_limits[env.action_ids]
            action = np.clip(raw_action, limits[:, 0], limits[:, 1])
            if np.max(np.abs(action[:7] - previous[:7])) > max_joint_jump:
                raise RuntimeError("joint_jump_abort")
            previous = action.copy()
            command[env.action_ids] = action
            closed_once |= bool(np.all(action[7:] < 0.0))
            active_release |= closed_once and bool(np.all(action[7:] > 0.3))
            encoder.stdin.write(np.concatenate(
                [images["head_cam"], images["camera_right"]], axis=1,
            ).tobytes())
            # Preserve the attempted action even if a physics substep aborts.
            error = None
            try:
                env.step(command)
            except RuntimeError as exc:
                error = exc
            obs = env.observe()
            max_lift = max(max_lift, float(obs["target"][2]) - initial_z)
            clearance, supported, released, speed = env.placement.measure(env)
            good = stable_frame(max_lift, clearance, supported, released, speed) and active_release
            stable = stable + 1 if good and error is None else 0
            rows.append({
                "state": state, "action": action, "action_raw": raw_action,
                "state_after": env.data.qpos[env.qids[env.action_ids]].copy(),
                "qpos_after": env.data.qpos.copy(), "target": obs["target"],
                "ee": obs["ee"], "released": released,
                "wall_clearance_m": clearance, "basket_floor_contact": supported,
                "object_speed": speed, "stable_steps": stable,
                "time": float(env.data.time), "complete_step": error is None,
            })
            if error:
                raise error
            if np.max(np.abs(env.data.qpos[env.left_qids] - env.start_qpos[env.left_qids])) > 0.02:
                raise RuntimeError("inactive_arm_abort")
            if obs["target"][2] < 0.5:
                raise RuntimeError("object_fell_abort")
            if stable >= 50:
                result.update(success=True, reason="stable_released_placement")
                break
    except Exception as exc:
        result.update(success=False, reason=str(exc))
    finally:
        encoder.stdin.close()
        code = encoder.wait(timeout=60)
        log.close()
        if code:
            result.update(success=False, reason=f"video_encoder_error:{code}")
        if rows:
            np.savez_compressed(folder / "trajectory.npz",
                                **{key: np.stack([r[key] for r in rows]) for key in rows[0]})
        if chunks:
            np.savez_compressed(folder / "predicted_chunks.npz", actions=np.stack(chunks))
        result.update(num_frames=len(rows), max_lift_m=max_lift,
                      stable_success_steps=stable, active_release=active_release,
                      image_mode=runtime.image_mode, action_processing="clip",
                      policy_actions_replaced=sum(
                          int(np.any(row["action_raw"] != row["action"])) for row in rows),
                      expert_takeover=False,
                      control_hz=50, initial_target_z=initial_z)
        if code == 0 and rows:
            probe = json.loads(subprocess.check_output([
                "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                "-show_entries", "stream=nb_read_frames", "-of", "json",
                str(folder / "cameras.mp4"),
            ], text=True))
            result["video_frames"] = int(probe["streams"][0]["nb_read_frames"])
            if result["video_frames"] != len(rows):
                result.update(success=False, reason="video_action_alignment_error")
        save(folder / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026110700)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--max-joint-jump", type=float, default=0.35)
    parser.add_argument("--image-mode", choices=("pad", "legacy-stretch"), default="pad")
    args = parser.parse_args()
    if (args.trials < 1 or args.max_steps < 1
            or not np.isfinite(args.max_joint_jump) or args.max_joint_jump <= 0):
        parser.error("Trial count, step count and jump limit must be positive")
    if args.output.exists():
        parser.error("Output already exists")
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "simulation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for port in (7070, 7072):
            with socket.socket() as connection:
                if connection.connect_ex(("127.0.0.1", port)) == 0:
                    raise RuntimeError(f"Camera port {port} is already in use")
        runtime = PolicyRuntime(args.checkpoint, args.tokenizer, args.image_mode)
        args.output.mkdir(parents=True)
        save(args.output / "provenance.json",
             provenance(args.checkpoint.resolve(), args.tokenizer.resolve()))
        env = EvaluationEnv()
        cameras = None
        try:
            save(args.output / "settings.json", {
                "scene_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
                "checkpoint_name": args.checkpoint.name, "trials": args.trials,
                "seeds": [args.seed + i for i in range(args.trials)],
                "image_mode": args.image_mode, "max_steps": args.max_steps,
                "max_joint_jump_rad": args.max_joint_jump,
                "fixed_scene": True, "action_processing": "clip", "expert_takeover": False,
                "action_limits": env.ctrl_limits[env.action_ids].tolist(),
                "success": "lift>8cm; full box inside; floor support; active release; speed<2cm/s for 50 steps",
                "guard": "clip actuator targets; abort on collision, joint jump or inactive-arm motion",
            })
            cameras = Cameras(env)
            results = []
            for i in range(args.trials):
                row = trial(env, cameras, runtime, args.output / f"trial_{i:03d}",
                            args.seed + i, args.max_steps, args.max_joint_jump)
                results.append(row)
                report = {"completed": len(results), "requested": args.trials,
                          "complete": len(results) == args.trials,
                          "successes": sum(r["success"] for r in results),
                          "success_rate": sum(r["success"] for r in results) / len(results),
                          "trials": results}
                save(args.output / "summary.json", report)
                print("TRIAL_RESULT", json.dumps(row), flush=True)
        finally:
            try:
                if cameras:
                    cameras.close()
            finally:
                env.close()


if __name__ == "__main__":
    main()
