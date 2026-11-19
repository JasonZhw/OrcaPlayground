"""视觉接地：模型给像素坐标 → 单应矩阵 → 桌面世界坐标。

模型只负责"点出物体中心"，后端用标定好的单应矩阵反算桌面 (x, y)，
再交给逆运动学。模型全程不接触坐标，坐标从图像 + 标定矩阵来。

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

    # Hartley 归一化：世界坐标动辄 30 量级，不归一化会把 DLT 解成垃圾。
    def _normalize(pts):
        c = pts.mean(axis=0)
        d = float(np.sqrt(((pts - c) ** 2).sum(axis=1)).mean())
        sc = np.sqrt(2.0) / d if d > 1e-12 else 1.0
        T = np.array([[sc, 0.0, -sc * c[0]], [0.0, sc, -sc * c[1]], [0.0, 0.0, 1.0]])
        return (pts - c) * sc, T

    pn, Tp = _normalize(pixels)
    wn, Tw = _normalize(world_xy)
    n = len(pn)
    A = np.zeros((2 * n, 8))
    b = np.zeros(2 * n)
    for i, ((u, v), (x, y)) in enumerate(zip(pn, wn, strict=False)):
        A[2 * i] = [u, v, 1, 0, 0, 0, -u * x, -v * x]
        A[2 * i + 1] = [0, 0, 0, u, v, 1, -u * y, -v * y]
        b[2 * i] = x
        b[2 * i + 1] = y
    h, *_ = np.linalg.lstsq(A, b, rcond=None)
    Hn = np.eye(3)
    Hn[0] = [h[0], h[1], h[2]]
    Hn[1] = [h[3], h[4], h[5]]
    Hn[2] = [h[6], h[7], 1.0]
    H = np.linalg.inv(Tw) @ Hn @ Tp
    return H / H[2, 2]


def pixel_to_world(H, u, v) -> tuple[float, float]:
    """像素 → 桌面坐标。"""
    q = H @ np.array([u, v, 1.0])
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
    for (u, v), (x, y) in zip(pixels, world_xy, strict=False):
        wx, wy = pixel_to_world(H, u, v)
        errs.append(np.hypot(wx - x, wy - y))
    return np.asarray(errs)


def residual(H, pixels, world_xy) -> float:
    """标定残差：把所有像素反算回世界，与真值的平均欧氏距离（米）。"""
    errs = []
    for (u, v), (x, y) in zip(pixels, world_xy, strict=False):
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


def calibrate(bridge, scene, env, arm, backend, object_key, ref_desc,
              x_range, y_range, nx=4, ny=4, table_z=None,
              out_path="layouts/homography.json", loop=None, samples=5) -> dict:
    """把参考物体沿桌面网格移动，让模型报像素，用 RANSAC 拟合单应矩阵。

    每个场景的相机位置不同，所以每个场景要标定一次；结果写到独立文件。
    ref_desc 是给模型看图的物体描述（如「黄色收纳筐」），必须能可靠指认。
    """
    from .main import FrameGrabber
    from .tools import _to_jpg

    xs = np.linspace(x_range[0], x_range[1], nx)
    ys = np.linspace(y_range[0], y_range[1], ny)
    if table_z is None:
        table_z = float(scene.body_pose(object_key)[0][2])
    pixels, world, records = [], [], []
    q0 = env.data.qpos.ravel().copy()  # 标定前位形，结束归位
    grabber = FrameGrabber(bridge=bridge, loop=loop)
    try:
        for i, (y, x) in enumerate([(y, x) for y in ys for x in xs]):
            addr = scene.free_joint_address(object_key)
            q = env.data.qpos.ravel().copy()
            q[addr:addr + 3] = [x, y, table_z]
            env.set_joint_qpos(q)
            env.mj_forward()
            frame_index = int(3000000 + i)
            if loop is not None:
                loop.run_until_complete(bridge.render(
                    env.data.qpos.ravel(), float(env.data.time),
                    simulate_index=frame_index, request_idr=True))
            # 只接受「这一步渲染出来的那一帧」；否则拿到上一步的旧帧，
            # 像素和世界坐标就错位，标定残差会飙到几十厘米。
            frames = grabber.grab(max_wait=8.0, required_simulate_index=frame_index)
            img = frames.get("head_cam")
            if img is None:
                continue
            h, w = img.shape[:2]
            prompt = (f"这是桌面俯视图（{w}x{h}）。找出「{ref_desc}」的中心。"
                      f"只能用 0~{w-1} 的 u 和 0~{h-1} 的 v。"
                      '只输出 JSON：{"u":整数,"v":整数}')
            # 模型像素有 ±10px 噪声，同一帧多问几次取中位数。
            us, vs = [], []
            jpg = _to_jpg(img)
            for _ in range(max(1, samples)):
                try:
                    raw = backend.ask([jpg], prompt)
                except Exception:  # noqa: BLE001
                    continue
                u, v = _parse_pixel(raw)
                if u is not None and 0 <= u < w and 0 <= v < h:
                    us.append(u)
                    vs.append(v)
            if not us:
                print(f"  [标定] ({x:+.3f},{y:+.3f}) 像素无效，跳过", flush=True)
                continue
            u = int(round(float(np.median(us))))
            v = int(round(float(np.median(vs))))
            pixels.append((u, v))
            world.append((x, y))
            records.append({"x": float(x), "y": float(y), "u": u, "v": v})
            spread = f"±{float(np.std(us)):.0f}/{float(np.std(vs)):.0f}px"
            print(f"  [标定] ({x:+.3f},{y:+.3f}) ← 像素 ({u},{v})  {spread}  n={len(us)}", flush=True)
    finally:
        grabber.close()
        # 把参考物放回原位，不留下被挪动的场景
        try:
            env.set_joint_qpos(q0)
            env.mj_forward()
            if loop is not None:
                loop.run_until_complete(bridge.render(
                    env.data.qpos.ravel(), float(env.data.time),
                    simulate_index=3999999, request_idr=True))
            print("  [标定] 参考物已归位", flush=True)
        except Exception:  # noqa: BLE001
            pass

    if len(pixels) < 4:
        raise RuntimeError(f"有效标定点不足（{len(pixels)}/4），检查抓图与模型像素输出")
    pix = np.array(pixels, dtype=float)
    wor = np.array(world, dtype=float)
    # 模型像素会有离群点，普通最小二乘会被带偏；用 RANSAC 剔掉再拟合。
    H, errs = fit_homography_ransac(pix, wor, inlier_thresh=0.05)
    inl = errs < 0.05
    res = float(errs[inl].mean()) if inl.any() else float(errs.mean())
    print(f"  [标定] 内点 {int(inl.sum())}/{len(errs)}，内点平均误差 {res * 1000:.1f} mm", flush=True)
    save_homography(H, out_path, meta={"object": object_key, "residual_m": res,
                                       "inliers": int(inl.sum()), "points": len(errs)})
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

