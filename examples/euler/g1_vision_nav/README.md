# G1 分层视觉导航 Demo

> 面向学员与讲师的完整实训课程见 [BOOTCAMP.md](BOOTCAMP.md)。

这个目录是 Demo 1 的新起点。机器人本体已从 Go2 切换为 Unitree G1；当前已完成 Layout 重建、固定 G1 生成、低层运控兼容验证、资产自带 Studio RGB 相机验证、点目标闭环和第一版 RGB 工作台避障。

## 已确认的本地资产

- OrcaLab G1 spawnable：`assets/cae3c6559556dd4f/default_project/prefabs/g1_pick_usda`
- 程序生成的 actor 名称：`g1_pick_usda`
- 运控参考 MJCF：`assets/g1/g1_29dof_camera.xml`（独立本地模型，不是固定 spawnable）
- 已验证的相机实体：`camera_head`
- 头部 RGB WebSocket：`7072`
- 左侧 RGB WebSocket：`7071`
- 右侧 RGB WebSocket：`7070`
- Depth WebSocket：尚未确认，当前禁用
- G1 低层速度控制模型：`assets/g1/models/dec_loco/model_6600.onnx`
- 低层模型配置：`assets/g1/config/g1_29dof_hist.yaml`
- 可复用的 Euler 运控参考：`examples/euler/07_locomotion/`
- 可复用的相机激活参考：`examples/euler/08_video_capture/`

固定 spawnable 的 MuJoCo 在线导出结果为 `ncam=0`，模型 XML 中不存在 `<camera>`，但这不能排除资产带有 Studio 侧 Camera Component。它包含 `head_camera1`、`head_camera2` 等相机相关 body；相机流以 OrcaLab Studio 的实际在线结果为准。验证程序不会再用 `ncam` 阻止连接。

在线抓帧已经确认 7070、7071、7072 都是独立的三通道 RGB 流，并不是“RGB + Depth”端口对。第一阶段正式使用 7072 头部 RGB；在找到并标定真正的深度端点前，不能用任何一路现有图像计算米制障碍物距离。

机器人 spawnable 已锁定，不再尝试其他 G1 候选路径。在线扫描已确认该资产包含 floating base、29 个标准 G1 关节和 45 个执行器：腿与腰共 15 个位置伺服、双臂共 14 个力矩电机、双手共 16 个位置伺服。现有 locomotion 策略直接从 floating base 和 29 个关节状态构造观测，不依赖旧资产中的 `imu_quat` / `imu_gyro` 传感器名称。

`g1_pick_usda` 的执行器类型与训练用 G1 资产不同，不能直接把 29 维目标关节角送入其位置伺服。`envs/euler/g1_vision_nav/g1_pick_locomotion_env.py` 会先计算策略要求的 PD 力矩，再利用资产公开的 `Kp/Kd` 反算腿和腰的位置指令；双臂仍直接接收力矩，双手保持零控制。

## 分层结构

```text
G1 camera_head RGB（约 30 Hz）
          │
          ▼
视觉导航上层（10 Hz）
RGB + 机器人坐标系目标 + 本体速度 + 上一条指令
          │
          ▼
(forward, lateral, yaw_rate, walk_enabled)
          │
          ▼
限幅 / 加速度限制 / 急停
          │
          ▼
冻结的 G1 locomotion ONNX（50 Hz）
          │
          ▼
29 关节目标 + 物理步 PD（1000 Hz）
```

上层只输出机体系速度指令，不直接输出 29 个关节动作。第一版冻结低层模型，不重新训练 G1 运控；这样视觉导航的成功或失败不会和步态训练混在一起。

## 当前代码边界

- `envs/euler/g1_vision_nav/contracts.py`：视觉观测、速度指令和导航策略接口。
- `envs/euler/g1_vision_nav/camera_stream.py`：实时 RGB 与可选 depth-preview 客户端。
- `envs/euler/g1_vision_nav/command_bridge.py`：上层指令限幅，并映射到现有 `G1Locomotion.set_commands()`。
- `envs/euler/g1_vision_nav/config.py`：相机端口、三层频率、初始安全速度和 G1 模型路径。
- `envs/euler/g1_vision_nav/point_goal_navigator.py`：世界坐标目标到机体系目标的变换，以及低速点目标基线控制器。
- `envs/euler/g1_vision_nav/visual_avoidance.py`：读取 7072 RGB、检测当前场景绿色工作台，并在点目标控制器外增加减速和带保持的局部绕行。
- `envs/euler/g1_vision_nav/g1_pick_locomotion_env.py`：把现有 29-DoF 运控策略适配到 G1 Pick 的 45 个混合执行器。
- `envs/euler/g1_vision_nav/g1_camera_validation_env.py`：相机资产预检、RGB 首帧/帧率/内容校验，同时维持原地站立。
- `envs/euler/g1_vision_nav/g1_vision_nav_env.py`：同步相机、点目标、10 Hz 上层控制、50 Hz 冻结运控、reward 与安全急停。
- `envs/euler/g1_vision_nav/g1_visual_avoidance_env.py`：记录视觉风险与介入次数，并要求在线测试确实看见障碍、改变指令且安全到达。
- `scene.yaml`：当前 Layout、障碍物范围、固定 G1 和出生点记录。
- `layout_scene.py`：读取 OrcaLab v3 Layout JSON，并发布 Layout 全部物体和固定 G1。
- `validate_locomotion.py`：暂不接视觉导航，只验证现有 G1 ONNX + PD 运控是否能站立和行走。
- `validate_camera.py`：激活固定 G1 的 7072 Studio 头部实时流，让仿真先 render，再异步验证并保存 RGB 样本。
- `run_point_goal.py`：运行自动点目标导航，可指定绝对目标或出生朝向前方的相对距离。
- `run_visual_avoidance.py`：运行 7072 RGB 工作台检测、局部绕行和冻结 G1 运控的完整闭环。

`G1VisionNavEnv` 已经完成第一阶段闭环：RGB 与导航状态同步进入 `NavigationObservation`，点目标基线暂时只使用相对目标和本体状态，尚未用 RGB 决策。下一阶段在同一个 `VisualNavigator` 接口下替换为视觉避障策略，无需修改底层 ONNX 运控。

## 先验证 G1 运控

当前 Layout 文件是 `/home/jason77/SY/demo1_v1.json`，包含 15 个场景物体，不包含 G1。脚本默认在障碍物东侧空地 `(-18.5, -13.5, 0)` 生成 G1，朝 `+x` 方向，首次前进会远离家具。

运行前先保存 Layout JSON，并启动 OrcaLab。脚本会清空 OrcaLab 当前运行场景，再从 JSON 重建场景，因此不要把尚未导出的临时修改留在编辑器里。

```bash
conda activate orca
cd /home/jason77/SY/OrcaPlayground
python examples/euler/g1_vision_nav/validate_locomotion.py
```

程序依次验证站立、前进、左转、左移和停止；每个阶段会在终端暂停，观察画面后按 Space 继续。也可以覆盖出生点：

```bash
python examples/euler/g1_vision_nav/validate_locomotion.py \
    --spawn-x -18.5 --spawn-y -13.5 --spawn-yaw 0
```

这一步的通过条件是 G1 不摔倒、基座高度维持在 `0.6–0.9m`、ONNX 输出有限且关节力矩没有持续触限。它不读取相机，也不测试导航。

2026-08-21 的完整 1000 步测试结果为 `82/82 passed`：站立、`0.5m/s` 前进、左转、`0.3m/s` 侧移和停止全部通过。基座高度约为 `0.729–0.790m`，最大俯仰/横滚倾角约 `0.084rad`，策略输出始终有限，统计到的力矩触限比例为 `0`。

## 相机验证

相机依赖已安装到 `orca` 环境：`av` 和 `opencv-python`。补好资产传感器后运行：

```bash
conda activate orca
cd /home/jason77/SY/OrcaPlayground
python examples/euler/g1_vision_nav/validate_camera.py
```

程序会自动尝试解析并激活固定 G1 的 Studio Camera Component，让 G1 原地站立约 10 秒，验证 RGB 分辨率、数据类型、非空画面和帧号增长，并把样本保存到 `/tmp/g1_camera_head_rgb.png`。

相机客户端在 `set_camera_sensor_info()` 后立即启动，但不会在第一个控制步之前阻塞等待图像；仿真会先持续执行 `step()` 和 `render()`，随后在循环中轮询首帧。这适配了由 Studio 渲染而非 MuJoCo `ncam` 直接提供的 Camera Component。

固定 G1 实际包含 `head_cam`、`camera_left`、`camera_right` 三路 Camera Component。2026-08-21 在线同时抓帧确认其原生映射为：`camera_head/head_cam → 7072`、`camera_left → 7071`、`camera_right → 7070`。激活时不能把 actor 下所有相机强制覆盖成同一个 WebSocket 端口；代码保留资产的原生端口分配，只连接 7072 头部 RGB。

7072 画面方向是正立的，但固定资产当前的头部相机俯角较大，主要覆盖双臂、脚和近处地面，远处障碍物与地平线基本不可见。当前 Demo 接受这一硬件视角，通过降低速度把它作为近场安全相机使用；约 1–2 米范围的工作台已经能稳定进入画面。若后续需要更高速导航、远距离规划或语义巡检，仍建议在资产侧增加真正的前视相机。

当前 OrcaGym Euler 公共接口只能激活相机，并修改分辨率、FOV、裁剪面和流端口，没有修改 Studio Camera Component 位置或旋转的接口。因此相机俯角不能在本 Demo 的 Python 代码里直接修正；必须把调整保存在可生成的资产/预制体中，或由 OrcaLab 后续提供可持久化的实例组件位姿接口。

## 点目标导航

第一阶段先验证上层目标控制与冻结运控之间的闭环。默认目标位于出生朝向前方 1 米，控制器先站立 2 秒等待相机，然后以最高 `0.15m/s` 前进、`0.08m/s` 横移和 `0.35rad/s` 转向。大角度时先原地转向，接近目标后使用前进、横移和转向的低速组合消除二维误差。

```bash
conda activate orca
cd /home/jason77/SY/OrcaPlayground
python examples/euler/g1_vision_nav/run_point_goal.py
```

指定世界坐标目标：

```bash
python examples/euler/g1_vision_nav/run_point_goal.py \
    --goal-x -18.5 --goal-y -12.7 --num-steps 900
```

程序同时检查基座高度、倾角、策略输出、力矩触限、RGB 帧增长和非足部环境碰撞。脚部支撑接触允许；跌倒或身体碰到环境物体会触发速度急停。

2026-08-21 在线结果：

- 前方点目标：初始距离 `1.323m`，最终 `0.310m`，`125/125 passed`。
- 左侧点目标：完成明显左转和弧线接近，最终 `0.313m`、最小 `0.296m`，`156/156 passed`。
- 两次测试均未跌倒、未检测到非足部环境碰撞，7072 RGB 全程连续产帧。

`run_point_goal.py` 本身不是视觉避障：RGB 虽已进入统一观测并在线验证，但 `PointGoalNavigator` 明确不读取像素。它用于排除世界坐标、坐标系转换、速度桥接和底层步态问题；下一节的 `run_visual_avoidance.py` 才会让 RGB 实际改变运动指令。

`--num-steps` 是 50 Hz 低层控制步数，因此仿真窗口约为 `num_steps / 50` 秒。增大它只会延长运行时间，不会提高速度；到达目标后 G1 会保持站立直到窗口结束。例如 `4000` 步约为 80 秒。

## RGB 工作台避障

第一版视觉闭环直接读取固定 G1 的 `camera_head → 7072` RGB。它在画面上方 72% 的近场区域检测当前 Layout 中的绿色工作台，比较画面左右占用率；达到风险阈值后会限制前进速度、向较空一侧横移并转向。工作台刚离开画面时，控制器仍保持约 6 秒的低速斜向绕行，避免立刻转回目标而擦碰桌边。

运行默认验证路线：

```bash
conda activate orca
cd /home/jason77/SY/OrcaPlayground
python examples/euler/g1_vision_nav/run_visual_avoidance.py --num-steps 4000
```

默认出生点仍为 `(-18.5, -13.5)`，目标为 `(-22.0, -15.7)`；直线路径会接近 `industrial_workbench_1`，因而能够验证 RGB 是否真实改变了指令。也可覆盖目标：

```bash
python examples/euler/g1_vision_nav/run_visual_avoidance.py \
    --goal-x -22.0 --goal-y -15.7 --num-steps 4000
```

2026-08-21 在线结果为 `655/655 passed`：初始目标距离 `3.864m`，最终 `0.309m`、最小 `0.293m`；视觉风险峰值 `0.0569`，视觉层介入 `282` 个 10 Hz 决策步；全程未跌倒、未发生非足部环境碰撞，7072 RGB 持续产帧。到达约发生在 step 2400，之后继续稳定站立到 step 4000。

这仍是面向当前 Layout 的可解释基线，不是通用障碍物感知：它可靠覆盖绿色工作台，但不能据此宣称已经识别白色桌腿、黑白货架、透明物体或任意新纹理。后续应保留这条确定性基线用于回归测试，同时采集 RGB、位姿、接触和动作数据，训练或接入类别无关的深度/占用感知上层。

## 资产中心选环境时记录这些信息

1. 环境资产卡片里的完整 `Path`，不是中文显示名或缩略图标题。
2. 导入/订阅后实际生成的 actor 名称。
3. 场景推荐出生位置和朝向；至少要有一个离障碍物安全的位置。
4. 可行走区域的大致 x/y 范围，以及地面高度。
5. 场景是否自带动态物体、楼梯、透明材质或无碰撞装饰物。

把前四项填入 `scene.yaml` 或直接发给开发者。确定性 RGB 工作台避障完成后，训练版视觉导航阶段依次完成：

1. 确定可行走边界、训练出生点和目标采样范围。
2. 为桌子、货架等障碍物确认可用于碰撞统计的 geom/body 名称。
3. 将 7072 RGB 缩放后与相对目标、本体速度和上一条指令组成训练观测。
4. 用类别无关的学习策略替换当前绿色工作台检测器，动作继续使用 `(forward, lateral, yaw_rate)`。
5. 先在稀疏障碍课程训练，再逐渐增加桌子密度、目标距离、纹理和光照变化。
6. 保留当前点目标控制器作为无障碍基线和回归测试，不重新训练底层 G1 步态。

## 上层训练的第一版定义

- 观测：缩放后的 RGB、目标在机器人坐标系的 `(x, y)`、本体平面速度、偏航角速度、上一条速度指令。
- 动作：`forward`、`lateral`、`yaw_rate` 三个连续量；到达目标或急停时关闭 walking。
- 正奖励：目标距离减少、到达目标。
- 负奖励：碰撞、跌倒、超时、动作突变、无进展。
- Curriculum：空旷目标跟随 → 稀疏静态障碍 → 密集障碍/拐角 → 纹理和光照随机化。

第一阶段不做语义巡检，也不做端到端图像到关节动作。等 point-goal 导航稳定后，再加入巡检点序列、目标识别和任务状态机。
