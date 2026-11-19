"""只保留双臂的任务模型（基于官方 mj_joint_strip）。

作用
----
把 OrcaLab 给的完整 G1 模型裁成「只剩两条手臂 + 两个夹爪」：
腿、腰、头、全部关节都被物理删除，底座也固定到世界。

为什么需要
----------
1. 指令层锁定（JointLock）只是"不发指令"，腿仍受重力下沉（实测漂移 62 mm）；
   物理删除后不存在被驱动的可能。
2. 关节数从 30 降到 16，执行器从 45 降到 20，物理步进更快。
3. 每次运行的状态完全确定，不会因为上一轮的残留姿态而结果不同。

渲染
----
OrcaStudio 需要完整维度的 qpos，所以渲染前用 QposBridge.pad() 补齐。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import xml_strip as strip

# 保留：左臂 7 + 右臂 7 + 两个夹爪
KEEP_BOTH_ARMS = (
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    "gripper_l_inner_joint", "gripper_l_outer_joint",
    "gripper_r_inner_joint", "gripper_r_outer_joint",
)


class ArmOnlyModel:
    """裁剪结果：精简 XML 路径 + 渲染用 qpos 补齐器 + 统计信息。"""

    def __init__(self, xml_path, out_path, info):
        self.xml_path = Path(xml_path)
        self.out_path = Path(out_path)
        self.info = info
        self.bridge = info["bridge"]

    def pad(self, qpos):
        """把精简模型的 qpos 补齐成 OrcaStudio 需要的完整维度。"""
        return self.bridge.pad(np.asarray(qpos, dtype=np.float64))

    def summary(self) -> str:
        i = self.info
        return (f"关节 {i['nq_full']}→{i['nq_stripped']}  "
                f"执行器 {i['nu_full']}→{i['nu_stripped']}  "
                f"body {i['nbody_full']}→{i['nbody_stripped']}  "
                f"保留 {len(i['kept'])} 关节，删除 {len(i['dropped'])} 个")


def build(xml_path, out_path=None, agent_name: str = "unitree_g1_1") -> ArmOnlyModel:
    """读取完整场景 XML，裁出只含双臂的模型。

    参数
    ----
    xml_path : OrcaLab 的世界模型路径（bridge.load_model_xml() 的返回值）
    out_path : 裁剪后写到哪里；默认与源文件同目录加 _armonly 后缀
    agent_name : 机器人前缀，如 unitree_g1_1
    """
    import mujoco

    xml_path = Path(xml_path)
    if out_path is None:
        out_path = xml_path.with_name(xml_path.stem + "_armonly.xml")
    out_path = Path(out_path)

    # 1) 载入完整模型，取关节名
    mj_full = mujoco.MjModel.from_xml_path(str(xml_path))
    md_full = mujoco.MjData(mj_full)
    mujoco.mj_forward(mj_full, md_full)
    names = [mujoco.mj_id2name(mj_full, mujoco.mjtObj.mjOBJ_JOINT, i) or ""
             for i in range(mj_full.njnt)]

    # 2) 决定删哪些
    drop, kept, foreign = strip.plan_strip(
        names, agent_name, keep=KEEP_BOTH_ARMS, keep_base=False
    )
    if not drop:
        raise RuntimeError("没有可裁剪的关节，检查 agent_name 是否正确")

    # 3) 改 XML：删关节/执行器/约束/传感器，并把被删 body 的位姿烘焙进父 body
    xml = xml_path.read_text()
    xml, report = strip.strip_xml(xml, drop)
    xml, baked = strip.bake_dropped_bodies(xml, mj_full, md_full, drop)
    out_path.write_text(xml)

    # 3.5) 关闭非手臂链（躯干/腿/头/底座）的碰撞体：保留视觉，但机械臂不再撞躯干
    keep_tokens = (
        "left_shoulder", "left_elbow", "left_wrist",
        "right_shoulder", "right_elbow", "right_wrist",
        "gripper_l", "gripper_r", "arm_l_end", "arm_r_end",
    )
    strip.disable_non_chain_collision(out_path, agent_name, keep_tokens)

    # 4) 载入精简模型并建立 qpos 补齐器
    mj_str = mujoco.MjModel.from_xml_path(str(out_path))
    bridge = strip.QposBridge(mj_full, md_full, mj_str)

    info = {
        "bridge": bridge,
        "kept": kept,
        "dropped": sorted(drop),
        "foreign": foreign,
        "baked": baked,
        "report": report,
        "nq_full": int(mj_full.nq), "nq_stripped": int(mj_str.nq),
        "nu_full": int(mj_full.nu), "nu_stripped": int(mj_str.nu),
        "nbody_full": int(mj_full.nbody), "nbody_stripped": int(mj_str.nbody),
    }
    return ArmOnlyModel(out_path, out_path, info)


def make_arm_only_env(arm_only: ArmOnlyModel, endpoint: str = "127.0.0.1:50051"):
    """用裁剪后的 XML 创建 SortingEnv。

    底座已被删除（骨盆固定到世界），所以跳过原版用来焊死浮动底座的 fix_base()。
    """
    from .execution import SortingEnv

    class ArmOnlyEnv(SortingEnv):
        def fix_base(self):  # noqa: D401 - 底座已不存在
            """裁剪后没有浮动底座，无需等式约束焊死。"""
            return

    return ArmOnlyEnv(str(arm_only.out_path), endpoint=endpoint)
