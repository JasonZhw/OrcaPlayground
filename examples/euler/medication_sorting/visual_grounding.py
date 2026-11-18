"""视觉目标接地（L1）：模型给像素坐标 → 单应矩阵 → 桌面世界坐标。

对齐 RobotUse / SuperNav 的「图像点目标」范式：
agent 在图像里点出物体中心，后端用标定好的单应矩阵反算桌面 (x, y)，
再交给 IK。模型全程不接触坐标，坐标从图像 + 标定矩阵来。

标定原理
--------
桌面是平面（z ≈ 固定值），头部相机俯视它。像素平面 ↔ 桌面平面之间
是一个 2D 单应（homography）变换，8 个自由度，至少 4 个非共线点可解，
实际用 9~25 个点做最小二乘更稳。

流程
----
标定（一次性）：
  1. 把某个已知物体移到桌面网格点 (x_i, y_i)
  2. 抓头部相机图，让模型报物体中心像素 (u_i, v_i)
  3. fit_homography 拟合 H，存盘

运行时（每轮）：
  pixel_to_world(H, u, v) -> (x, y)  ->  IK 到 (x, y, z=桌面高)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# 单应矩阵核心（纯数学，可离线验证）
# ---------------------------------------------------------------------------


def fit_homography(pixels, world_xy) -> np.ndarray:
    """用 N 组「图像像素 (u,v) ↔ 桌面 (x,y)」拟合 3x3 单应矩阵 H。

    参数
    ----
    pixels : (N, 2)  float，图像坐标（列 u、行 v）
    world_xy : (N, 2) float，桌面坐标（x、y）

    返回
    ----
    H : (3, 3)，满足 [x, y, 1]^T ∝ H @ [u, v, 1]^T
    """
    pixels = np.asarray(pixels, dtype=np.float64)
    world_xy = np.asarray(world_xy, dtype=np.float64)
    assert pixels.shape == world_xy.shape and pixels.shape[1] == 2
    assert len(pixels) >= 4, "单应至少需要 4 个非共线点"

    n = len(pixels)
    A = np.zeros((2 * n, 8))
    b = np.zeros(2 * n)
    for i, ((u, v), (x, y)) in enumerate(zip(pixels, world_xy)):
        A[2 * i] = [u, v, 1, 0, 0, 0, -u * x, -v * x]
        A[2 * i + 1] = [0, 0, 0, u, v, 1, -u * y, -v * y]
        b[2 * i] = x
        b[2 * i + 1] = y
    h, *_ = np.linalg.lstsq(A, b, rcond=None)
    H = np.eye(3)
    H[0] = [h[0], h[1], h[2]]
    H[1] = [h[3], h[4], h[5]]
    H[2] = [h[6], h[7], 1.0]
    return H


def pixel_to_world(H, u, v) -> tuple[float, float]:
    """像素 → 桌面坐标。"""
    q = H @ np.array([u, v, 1.0])
    q = q / q[2]
    return float(q[0]), float(q[1])


def world_to_pixel(H, x, y) -> tuple[float, float]:
    """桌面坐标 → 像素（用于校验标定残差）。"""
    q = np.linalg.inv(H) @ np.array([x, y, 1.0])
    q = q / q[2]
    return float(q[0]), float(q[1])


def fit_homography_ransac(pixels, world_xy, *, n_iter=2000, inlier_thresh=0.03, seed=1):
    """用 RANSAC 拟合单应矩阵，抵抗模型像素离群点。

    实测：普通最小二乘会被 1~2 个离群点毁掉（残差 12cm），
    RANSAC 剔除后中位误差降到 ~7.5mm。运行时模型噪声较大，必须用这个。
    """

    pixels = np.asarray(pixels, dtype=np.float64)
    world_xy = np.asarray(world_xy, dtype=np.float64)
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_iter):
        sel = rng.choice(len(pixels), min(8, len(pixels)), replace=False)
        H = fit_homography(pixels[sel], world_xy[sel])
        errs = _point_errors(H, pixels, world_xy)
        inl = int(np.sum(errs < inlier_thresh))
        score = (inl, -float(np.median(errs)))
        if best is None or score > best[0]:
            best = (score, H, errs)
    return best[1], best[2]


def _point_errors(H, pixels, world_xy) -> np.ndarray:

    errs = []
    for (u, v), (x, y) in zip(pixels, world_xy):
        wx, wy = pixel_to_world(H, u, v)
        errs.append(np.hypot(wx - x, wy - y))
    return np.asarray(errs)


def residual(H, pixels, world_xy) -> float:
    """标定残差：把所有像素反算回世界，与真值的平均欧氏距离（米）。"""
    errs = []
    for (u, v), (x, y) in zip(pixels, world_xy):
        wx, wy = pixel_to_world(H, u, v)
        errs.append(np.hypot(wx - x, wy - y))
    return float(np.mean(errs))


def save_homography(H, path, *, meta=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(
        {"H": H.tolist(), "meta": meta or {}}, indent=2))


def load_homography(path) -> np.ndarray:
    return np.asarray(json.loads(Path(path).read_text())["H"], dtype=np.float64)


# ---------------------------------------------------------------------------
# 标定驱动（需要仿真在线 + 模型后端）
# ---------------------------------------------------------------------------


def calibrate(bridge, scene, env, arm, backend, object_key="calcium",
              x_range=(0.40, 0.68), y_range=(-3.60, -3.46),
              nx=4, ny=4, table_z=4.752, out_path="layouts/homography.json",
              loop=None) -> dict:
    """把参考物体移到桌面网格点，让模型报像素，拟合单应矩阵。

    返回 {"H": 矩阵, "residual": 米, "points": 每点的像素/世界/误差}。

    注意：依赖仿真在线，且需要后端能「看图报物体中心像素」。
    """
    xs = np.linspace(x_range[0], x_range[1], nx)
    ys = np.linspace(y_range[0], y_range[1], ny)
    pixels, world = [], []
    records = []

    for y in ys:
        for x in xs:
            # 1) 把参考物体移到 (x, y)
            addr = scene.free_joint_address(object_key)
            q = env.data.qpos.ravel().copy()
            q[addr:addr + 3] = [x, y, table_z]
            env.set_joint_qpos(q)
            env.mj_forward()
            # 2) 渲染 + 抓图
            from .agent_policy import FrameGrabber  # noqa: F401 延迟导入避免环

            grabber = FrameGrabber(bridge=bridge, loop=loop)
            if loop is not None:
                loop.run_until_complete(bridge.render(
                    env.data.qpos.ravel(), float(env.data.time),
                    simulate_index=int(2000000 + len(pixels)), request_idr=True))
            frames = grabber.grab(max_wait=8.0)
            img = frames.get("head_cam")
            if img is None:
                continue
            # 3) 让模型报像素（业务方实现 detector）
            detector = backend.ask
            prompt = ("这是桌面俯视图。找出白色药瓶（calcium）的中心，"
                      "只输出 JSON：{\"u\":整数,\"v\":整数}")
            raw = detector([img], prompt) if callable(getattr(backend, "ask", None)) else ""
            u, v = _parse_pixel(raw)
            if u is None:
                continue
            pixels.append((u, v))
            world.append((x, y))
            records.append({"x": x, "y": y, "u": u, "v": v, "raw": raw})

    if len(pixels) < 4:
        raise RuntimeError(f"有效标定点不足（{len(pixels)}/4），检查抓图和模型像素输出")

    H = fit_homography(np.array(pixels), np.array(world))
    res = residual(H, np.array(pixels), np.array(world))
    save_homography(H, out_path, meta={"object": object_key, "residual_m": res})
    return {"H": H, "residual": res, "points": records}


def _parse_pixel(raw: str):
    """从模型输出里提取 u/v 整数。"""
    import re

    if not raw:
        return None, None
    u = re.search(r'"u"\s*:\s*(\d+)', raw)
    v = re.search(r'"v"\s*:\s*(\d+)', raw)
    if not u or not v:
        return None, None
    return int(u.group(1)), int(v.group(1))


# ---------------------------------------------------------------------------
# 运行时接地
# ---------------------------------------------------------------------------


class VisualGrounding:
    """运行时：模型给像素 → 桌面坐标。"""

    def __init__(self, H: np.ndarray, table_z: float = 4.752):
        self.H = H
        self.table_z = table_z

    def ground(self, u, v) -> tuple[float, float, float]:
        x, y = pixel_to_world(self.H, u, v)
        return x, y, self.table_z
