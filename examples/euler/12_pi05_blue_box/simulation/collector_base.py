"""Shared contact checks and episode storage using Euler public APIs."""

import json

import numpy as np

from task_config import TARGET


class Collector:
    def hold(self, count=50):
        for _ in range(count):
            self.step()

    def target_contacts(self):
        geoms = self.env.model.get_geom_dict()
        target_ids = {item["GeomId"] for item in geoms.values() if item["BodyName"] == TARGET}
        bodies = {item["GeomId"]: item["BodyName"] for item in geoms.values()}
        contacts = []
        for contact in self.env.query_contact_simple():
            first, second = contact["geom1"], contact["geom2"]
            if first in target_ids or second in target_ids:
                contacts.append(bodies[second if first in target_ids else first])
        return contacts

    def result(self, initial_z):
        poses = self.env.get_body_xpos_xmat_xquat([TARGET, self.bin_body])
        target, basket = poses[TARGET], poses[self.bin_body]
        center = target["xpos"] + np.asarray(target["xmat"]).reshape(3, 3) @ [0, 0, 0.051722717]
        local = np.asarray(basket["xmat"]).reshape(3, 3).T @ (center - basket["xpos"])
        joint = "intel_core_i5_processor_package_1_3C_intelcore_joint"
        offset = self.env.joints[joint]["QvelIdxStart"]
        speed = float(np.linalg.norm(self.env.data.qvel[offset:offset + 3]))
        lift = float(self.max_target_height - initial_z)
        success = bool(abs(local[0]) < 0.20 and abs(local[1]) < 0.12
                       and 0.025 < local[2] < 0.16 and speed < 0.02 and lift > 0.08)
        return {
            "success": success, "center_in_bin_frame": local.tolist(),
            "object_speed": speed, "lift_height": lift,
            "max_target_height": float(self.max_target_height),
        }

    def save(self, outdir, result):
        if self.frames:
            np.savez_compressed(
                outdir / "trajectory.npz",
                **{key: np.stack([row[key] for row in self.frames]) for key in self.frames[0]},
            )
        (outdir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    def close(self):
        try:
            if self.cameras:
                self.cameras.close()
        finally:
            self.env.close()
