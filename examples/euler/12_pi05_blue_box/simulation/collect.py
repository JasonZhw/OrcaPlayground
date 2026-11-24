"""Record 50 Hz observations and actions with physical contact acceptance."""
import argparse
import hashlib
import json
import shutil
import subprocess
import time
import traceback

import numpy as np
from scipy.spatial.transform import Rotation

from environment import DualArmEnv, ROBOT
from build_model import OUTPUT

from camera_capture import Cameras
from collector_base import Collector
from planner_v9 import ArmPlanner
from task_config import ROOT

GRASP_ROTATION = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]], dtype=float)


def save_json(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2))
    temp.replace(path)


class DualCollector(Collector):
    def __init__(self, offline=False, center_offset=0.0, lateral_offset=-0.035):
        self.env = DualArmEnv(offline=offline)
        self.planner = ArmPlanner()
        self.cameras = None if offline else Cameras(self.env)
        self.offline = offline
        self.center_offset = center_offset
        self.lateral_offset = lateral_offset
        self.command = self.env.setpoint.copy()
        self.bin_body = next(n for n in self.env.model.get_body_names() if n.endswith("basket_01"))
        self.frames = []
        self.phase = "reset"
        self.video = None
        self.transport_active = False
        self.audit = []
        self.snapshots = False
        self.contact_hold_steps = 30

    def step(self):
        if shutil.disk_usage(ROOT).free < 2 * 1024**3:
            raise RuntimeError("Less than 2GiB free; stopping collection safely")
        before = self.env.observe()
        state = self.env.data.qpos[self.env.qids[self.env.action_ids]].copy()
        action = self.command[self.env.action_ids].copy()
        if self.cameras:
            images = self.cameras.capture()
            for name, image in images.items():
                if image.shape != (480, 640, 3) or image.std() < 2:
                    raise RuntimeError(f"Invalid synchronized camera: {name}")
            self.video.stdin.write(np.concatenate([images["head_cam"], images["camera_right"]], axis=1).tobytes())
        after, *_ = self.env.step(self.command)
        self.max_target_height = max(self.max_target_height, float(after["target"][2]))
        contacts = self.target_contacts()
        bilateral = all(any(f"gripper_r_{side}" in n for n in contacts)
                        for side in ("inner", "outer"))
        rotation = self.env.data.site_xmat(f"{ROBOT}_ee_center_site_r").reshape(3, 3)
        relative = rotation.T @ (after["target"] - after["ee"])
        if self.transport_active:
            self.audit.append({"bilateral": bilateral,
                               "relative_excursion": float(np.linalg.norm(relative - self.transport_reference))})
        self.frames.append({
            "state": state, "action": action, "target": before["target"],
            "time": float(self.env.data.time - 0.02), "phase": self.phase,
            "ee_after": after["ee"].copy(), "bilateral_after": bilateral,
            "relative_after": relative.copy(),
        })
        self.step_index += 1
        if self.step_index % 100 == 0:
            print("FRAME", self.step_index, self.phase, flush=True)
        if np.linalg.norm(after["pelvis"] - self.env.fixed_pelvis) > 1e-8:
            raise RuntimeError("Pelvis moved")
        left_error = np.max(np.abs(self.env.data.qpos[self.env.left_qids]
                                  - self.env.start_qpos[self.env.left_qids]))
        if left_error > 0.02:
            raise RuntimeError(f"Inactive left arm moved: {left_error}")
        if abs(after["target"][0] - 30.1) > 1 or after["target"][2] < 0.5:
            raise RuntimeError("Object left table area")
        return after

    def grip(self, angle):
        ids = self.env.finger_aids
        start = self.command[ids].copy()
        for fraction in np.linspace(0, 1, 60):
            self.command[ids] = start + fraction * (angle - start)
            self.step()
        self.hold(30)

    def move(self, name, position, rotation=GRASP_ROTATION):
        self.phase = name
        trajectory = self.planner.plan(self.env.data.qpos[self.env.arm_qids], position, rotation)
        if name in ("transfer", "align_basket", "place"):
            sample = np.linspace(0, len(trajectory) - 1, int(len(trajectory) * 1.3))
            trajectory = np.stack([np.interp(sample, np.arange(len(trajectory)), trajectory[:, j])
                                   for j in range(7)], axis=1)
        for waypoint in trajectory:
            self.command[self.env.arm_aids] = waypoint
            self.step()
        self.hold(25)
        error = float(np.linalg.norm(self.env.observe()["ee"] - position))
        actual_rot = self.env.data.site_xmat(f"{ROBOT}_ee_center_site_r").reshape(3, 3)
        angle = float(Rotation.from_matrix(actual_rot.T @ rotation).magnitude())
        print("REACHED", name, "error_m", error, "rotation_error_rad", angle, flush=True)
        if error > 0.01 or angle > 0.10:
            raise RuntimeError(f"Tracking error {name}: {error}m, {angle}rad")

    def run_episode(self, outdir):
        self.env.restore()
        self.command = self.env.setpoint.copy()
        self.frames, self.audit = [], []
        self.step_index = 0
        self.max_target_height = -np.inf
        self.transport_active = False
        self.outdir = outdir
        outdir.mkdir(parents=True, exist_ok=False)
        log = None
        if not self.offline:
            log = (outdir / "video.log").open("wb")
            self.video = subprocess.Popen([
                "ffmpeg", "-nostdin", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1280x480", "-r", "50",
                "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast",
                "-crf", "18", "-pix_fmt", "yuv420p", str(outdir / "cameras.mp4"),
            ], stdin=subprocess.PIPE, stderr=log)
        close_error = angle = None
        initial_z = float(self.env.observe()["target"][2])
        started = time.monotonic()
        result = {"success": False}
        try:
            self.phase = "open"
            self.grip(0.5)
            target = self.env.observe()["target"].copy()
            grasp = target + [0, 0, 0.115]
            self.move("pregrasp", grasp + [0, 0, 0.20])
            self.move("approach", grasp)
            self.phase = "center_settle"
            self.hold(25)
            obs = self.env.observe()
            close_error = float(np.linalg.norm(obs["ee"] - (obs["target"] + [0, 0, 0.115])))
            rot = self.env.data.site_xmat(f"{ROBOT}_ee_center_site_r").reshape(3, 3)
            angle = float(Rotation.from_matrix(rot.T @ GRASP_ROTATION).magnitude())
            if close_error > 0.005 or angle > 0.0872665:
                raise RuntimeError("Centered vertical closure gate failed")
            self.phase = "close"
            self.grip(-0.7)
            self.phase = "contact_hold"
            self.hold(self.contact_hold_steps)
            if not all(r["bilateral_after"] for r in self.frames[-25:]):
                raise RuntimeError("Bilateral contact not sustained")
            self.transport_reference = self.frames[-1]["relative_after"].copy()
            self.transport_active = True
            self.move("lift", target + [0, 0, 0.27])
            self.phase = "lift_hold"
            self.hold(30)
            if self.env.observe()["target"][2] - initial_z < 0.08:
                raise RuntimeError("Insufficient lift")
            basket = self.env.get_body_xpos_xmat_xquat([self.bin_body])[self.bin_body]
            # Compensate measured in-hand offset to target the object's center.
            center_offset_world = GRASP_ROTATION @ self.transport_reference + [0, 0, 0.051722717]
            place = np.asarray(basket["xpos"]) + np.asarray(basket["xmat"]).reshape(3, 3) @ [self.center_offset, self.lateral_offset, 0.15]
            place -= center_offset_world
            safe_z = max(0.98, float(place[2]) + 0.12)
            self.move("transfer", np.array([place[0], -6.900, safe_z]))
            self.move("align_basket", np.array([place[0], place[1], safe_z]))
            self.move("place", place)
            self.transport_active = False
            excursion = max(v["relative_excursion"] for v in self.audit)
            fraction = float(np.mean([v["bilateral"] for v in self.audit]))
            if excursion > 0.01 or fraction < 0.98:
                raise RuntimeError("Transport contact/slip gate failed")
            self.phase = "release"
            self.grip(0.5)
            self.move("retreat", np.array([place[0], place[1], safe_z + 0.05]))
            self.phase = "settle"
            self.hold(100)
            result = self.result(initial_z)
            result["released"] = not any("gripper_r_" in n for n in self.target_contacts())
            result["success"] = bool(result["success"] and result["released"])
            result["placement_center_offset_m"] = float(np.linalg.norm(result["center_in_bin_frame"][:2]))
        except Exception as exc:
            traceback.print_exc()
            result = {"success": False, "error": str(exc)}
        finally:
            self.transport_active = False
            if self.video:
                self.video.stdin.close()
                code = self.video.wait(timeout=60)
                log.close()
                self.video = None
                if code:
                    result.update(success=False, error=f"Video encoder failed: {code}")
            result.update(
                close_error_m=close_error, close_rotation_error_rad=angle,
                max_relative_excursion_m=max((v["relative_excursion"] for v in self.audit), default=None),
                bilateral_transport_fraction=float(np.mean([v["bilateral"] for v in self.audit])) if self.audit else None,
                num_frames=len(self.frames), control_hz=50, video_hz=50,
                image_action_alignment="one pre-action camera pair for each action",
                model_sha256=hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
                base_condition="fixed torso; both arms retained; right arm commanded",
                controller="cuRobo demonstrator; not pi05 policy",
                position_jitter_m=[0, 0], center_target_offset_m=self.center_offset,
                lateral_target_offset_m=self.lateral_offset,
                recorded=not self.offline, wall_seconds=time.monotonic()-started,
            )
            self.save(outdir, result)
        print("EPISODE_RESULT", json.dumps(result), flush=True)
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--dataset", default="blue_box_v9")
    parser.add_argument("--center-offset", type=float, default=0.0)
    parser.add_argument("--lateral-offset", type=float, default=-0.035)
    args = parser.parse_args()
    root = ROOT / "datasets" / args.dataset
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    records = [json.loads(p.read_text()) for p in sorted(raw.glob("episode_*/result.json"))]
    if any(not r.get("recorded") for r in records) and not args.offline:
        raise RuntimeError("Cannot mix offline diagnostics with recorded demonstrations")
    accepted = sum(r["success"] for r in records)
    if accepted >= args.count:
        return
    collector = DualCollector(args.offline, args.center_offset, args.lateral_offset)
    failures = 0
    try:
        for attempt in range(len(records), args.count + 10):
            result = collector.run_episode(raw / f"episode_{attempt:03d}")
            records.append(result)
            accepted += int(result["success"])
            failures = 0 if result["success"] else failures + 1
            save_json(root / "status.json", {
                "target": args.count, "accepted": accepted, "attempts": len(records),
                "complete": accepted >= args.count, "recorded": not args.offline,
                "trials": records,
            })
            if accepted >= args.count:
                break
            if failures >= 2:
                raise RuntimeError("Two failed demonstrations; diagnose before continuing")
        if accepted < args.count:
            raise RuntimeError("Accepted episode target not met")
    finally:
        collector.close()


if __name__ == "__main__":
    main()
