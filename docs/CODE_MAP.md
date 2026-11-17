# 代码功能地图（供微调参考）

> 标注 ⭐ 的是常调参数所在位置。

---

## 一、`agent_policy.py` — 决策循环

### 数据层

| 函数 / 类 | 作用 | 微调点 |
|---|---|---|
| `_bootstrap_path()` | 自动把仓库根加入 `sys.path`，免设 PYTHONPATH | |
| `Observation` | 一轮的观测：图 + 已入筐 + 待处理 + 上次错误 + 已试角度 | `to_prompt()` ⭐ 改提示词就在这里 |
| `Decision` | 模型输出结构：`action / object / yaw / reason` | `parse()` 容错解析 |
| `prepare_image()` ⭐ | 图片缩到 512px + JPEG q75 | `MAX_IMAGE_SIDE` / `JPEG_QUALITY` 控制延迟 |

### 模型后端

| 类 | 作用 |
|---|---|
| `AnthropicBackend` ⭐ | 主用。`ask()` 通用多图问答，`decide()` 决策 |
| `OpenAICompatBackend` | 兼容 OpenAI 协议的备用后端 |
| `HeuristicBackend` | 无模型时的规则兜底（按顺序抓） |
| `DryRunBackend` | 不联网的干跑，用于测流程 |
| `Verifier` ⭐ | 视觉校验「是否入筐」，`_verify_prompt()` 是提问模板 |

### 感知与执行辅助

| 类 / 函数 | 作用 | 微调点 |
|---|---|---|
| `FrameGrabber` ⭐ | 开三路相机流、抓帧 | `CAMERA_PORTS` 端口；`grab(max_wait)` 超时 |
| `arm_only_model.build()` ⭐ | 裁掉腿/腰/头/底座，只留双臂 | `KEEP_BOTH_ARMS` 白名单 |
| `arm_only_model.make_arm_only_env()` | 用裁剪 XML 建 `SortingEnv`，跳过 `fix_base` | |
| `_reset_to_demo_default()` ⭐ | 每轮平滑复位手臂到全 0 | `settle` 步数、smoothstep 过渡长度 |
| `_prepare_ready_pose()` | 仅碰撞时抬臂的兜底（默认不调用） | |
| `JointLock` | 指令层锁定非手臂关节（裁剪模式下不用） | |

### 主循环

| 方法 | 作用 |
|---|---|
| `AgentPolicy.run_sim_episode()` ⭐⭐ | **核心**：拍照→决策→执行→校验→记录 |
| `AgentPolicy._decide()` | 调模型，失败重试 2 次后退回启发式 |
| `AgentPolicy._verify()` | 视觉校验，失败返回 None 让流程走真值 |
| `AgentPolicy._record()` | 写一条日志记录 |
| `AgentPolicy.run_dry()` | 不连仿真的干跑 |
| `selfcheck()` ⭐ | 7 项环境自检 |

### 输出与进度

| 类 | 作用 |
|---|---|
| `Reporter` ⭐ | 终端分阶段进度（时间戳 + 已用时长） |
| `Metrics` | 统计 LLM 调用次数、延迟、失败、兜底 |

---

## 二、`workflow.py` — 任务语义与动作序列

| 名称 | 作用 | 微调点 |
|---|---|---|
| `OBJECT_KEYS` | 四件物体名 | 增删物体 |
| `DESTINATIONS` ⭐ | 物体 → 目标筐 | 改分拣规则 |
| `RELEASE_BOTTOM_CLEARANCE` ⭐ | 放入筐时离筐底的高度 | 按筐深调 |
| `HOVER_RIM_CLEARANCE` ⭐ | 搬运时高出筐沿多少 | |
| `BOTTLE_CARRY_EXTRA_HEIGHT` | 瓶子额外抬升 3cm | |
| **`MOTION`** ⭐⭐ | **8 个阶段的 (速度, 加速度, 沉降帧)** | **动作快慢、稳不稳，全在这里** |
| `grasp_target()` ⭐ | 算抓取位姿：`position / rotation / approach` | 握点高度比例、接近偏移 |
| `placement_targets()` ⭐ | 算放筐位姿：`above / release` | `descent` 上限（筐深时保护） |
| `Scene` | 场景封装：`body_pose` / `end_pose` / `collision_free` / `free_joint_address` | |
| `Trial.prepare_pick()` ⭐ | 抓前：抬 8cm + 张爪 | 抬升高度 |
| `Trial.pick_available()` | 选可抓的物体 | |
| **`Trial.pick_place()`** ⭐⭐ | **完整抓放序列（12 步）** | 各阶段顺序与参数 |
| `Trial.lower_for_release()` | 放入筐内，失败则从上方释放兜底 | |
| `placement_satisfied()` | 真值判定是否入筐 | 判定容差 |

---

## 三、`planning.py` — 几何

| 名称 | 作用 | 微调点 |
|---|---|---|
| `solve_ik()` ⭐ | IK：18 次随机重启，容差 3mm | `attempts`、容差 |
| `collision_free()` ⭐ | 碰撞检查 | `penetration_tolerance` |
| `servo_bias_feasible()` | 伺服力矩余量检查 | |
| `plan_motion()` | 关节空间规划（RRT） | |

---

## 四、`execution.py` — 伺服与夹具

| 名称 | 作用 | 微调点 |
|---|---|---|
| `SortingEnv.__init__` | 建环境、读执行器、`fix_base()` | |
| `SortingEnv.fix_base()` | 用等式约束焊死浮动底座 | 裁剪模型时被覆盖跳过 |
| `SortingEnv.step()` ⭐ | 发指令，含 bias 前馈补偿 | `frame_skip`、`timestep` |
| `Executor.move()` ⭐ | 走上一条轨迹（含 `max_velocity`/`max_acceleration`） | 默认 0.4/0.8（已被 MOTION 覆盖） |
| `Executor.hold()` | 原地保持 n 帧 | |
| `Executor.ramp_gripper()` | 夹爪渐变（避免瞬间闭合） | |
| `Grippers.set()` / `apply()` | 夹爪目标值换算 | |

---

## 五、`arm_only_model.py` — 关节裁剪（新增）

| 名称 | 作用 |
|---|---|
| `KEEP_BOTH_ARMS` ⭐ | 保留白名单：14 臂关节 + 4 夹爪关节名（按子串匹配，实际匹配到 30 个自由度） |
| `build(xml_path)` | 两步：`plan_strip` 决定删谁 → `strip_xml` 删 → `bake_dropped_bodies` 烘焙位姿 |
| `ArmOnlyModel.pad()` | 渲染前把 99 维 qpos 补齐到 121 维 |
| `make_arm_only_env()` | 建 `SortingEnv` 子类，覆盖 `fix_base()`（底座已不存在） |
| `summary()` | 打印裁剪统计 |

**效果**：`qpos 121→99`、执行器 `45→30`、非手臂关节漂移恒为 0、每次运行完全确定。

---

## 六、抖动问题的定位结果

| 抖动来源 | 原因 | 状态 |
|---|---|---|
| **启动瞬间** | `env.command[臂关节] = 0.0` 是瞬时跳变 | ✅ 改为 smoothstep 40 步过渡 |
| **动作过快** | `move()` 默认 0.4 m/s，比示范快 5–10 倍 | ✅ 用 `MOTION` 降到 0.03–0.10 |
| **腿部下沉** | 腿关节在重力下漂移（实测 62 mm） | ✅ 裁剪模型后物理删除 |
| **每轮起始姿态不同** | 场景不复位，继承上一轮的姿态 | ✅ 每轮平滑复位到全 0 |

---

## 七、常调参数速查

```python
# 动作快慢（workflow.py）
MOTION = {
    "clearance": (0.06, 0.12, 20),
    "approach":  (0.08, 0.15, 30),
    "descend":   (0.04, 0.08, 30),   # 想更慢就调这个
    "lift":      (0.05, 0.10, 30),
    "transfer":  (0.10, 0.20, 30),
    "lower":     (0.03, 0.06, 20),
    "withdraw":  (0.04, 0.08, 20),
    "return":    (0.08, 0.15, 30),
}

# 抓取角度候选（agent_policy.py）
YAW_CANDIDATES = (0, 15, -15, 30, -30, 45, -45, 60, -60, 75, -75, 90, -90)

# 图像压缩（agent_policy.py）
MAX_IMAGE_SIDE = 512
JPEG_QUALITY = 75

# IK 强度（planning.py 调用处）
solve_ik(..., attempts=18)      # 调大更稳，调小更快
```
