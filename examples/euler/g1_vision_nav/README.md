# ORCA Lab 实训教案：从 Layout 搭建到 G1 视觉巡检

> 课程状态：绿色桌子最小闭环和工厂左线巡检已经完成在线验证；工厂右线作为后续扩展。

## 一、实训目标与两个阶段

### 1.1 两阶段任务

本实训不会一开始就把 G1 放进复杂工厂并要求它完成全部巡检，而是通过两个由简到繁的
场景逐步验证系统能力：

1. **绿色桌子最小验证**：先验证 G1 运控、实时定位、RGB 取流、颜色检测、局部绕障和
   避障后的目标重跟随；
2. **工厂电气柜巡检**：在最小闭环成立后，再加入多个 waypoint、复杂环境和柜前拍照。

目前绿色桌子和工厂左侧路线已经完成在线验证；工厂右侧路线保留为后续扩展。

这门实训不从“重新训练一套步态”开始，而是复用已经能够稳定行走的 G1 运控模型，
在它上面开发视觉导航。课程最终完成一条清晰的分层链路：

```text
ORCA Lab Layout
环境、灯光、桌子、货架、电气柜
               │
               ▼
代码通过 AddActor 生成 g1_navi
并注册 Studio Camera Component
               │
        ┌──────┴──────┐
        ▼             ▼
camera_head RGB    G1 实时位姿
WebSocket 7072     gRPC 50051
        └──────┬──────┘
               ▼
视觉导航上层：目标跟随 + RGB 近场避障
               │
               ▼
(vx, vy, yaw_rate, walk_enabled)
               │
               ▼
冻结的 G1 locomotion ONNX
               │
               ▼
29 个本体关节目标 + 混合执行器适配
               │
               ▼
G1 行走、到达巡检点并保存 RGB 照片
```

课程重点不是让视觉网络直接控制关节，而是理解机器人系统如何分层：

- ORCA Lab 负责场景编辑、渲染与物理仿真；
- OrcaGym 负责 Python 与仿真服务之间的通信；
- OrcaPlayground 中的任务代码负责相机、导航、运控和验收逻辑；
- 上层导航只给速度指令，底层模型负责把速度变成稳定步态。

### 1.2 学习目标

完成实训后，学员应能够：

1. 在 ORCA Lab 中新建并保存一个可运行的 Layout；
2. 解释为什么本课程的带相机 G1 由代码通过 `AddActor` 生成；
3. 理解 gRPC 控制端口和 RGB WebSocket 端口的不同职责；
4. 复用官方 Lesson 7 的 G1 ONNX 运控模型；
5. 理解 29 个策略关节如何适配到 G1 Pick 的混合执行器；
6. 实现世界坐标目标、实时位姿和机体系速度指令之间的转换；
7. 让 RGB 颜色占比真正改变导航指令；
8. 依次完成绿色桌子验证和工厂左线巡检；
9. 区分“已在线验证的能力”和“后续计划”。

### 1.3 运行准备

#### 软件与仓库

```bash
conda activate orca
cd /home/jason77/SY/OrcaPlayground
```

需要准备：

- ORCA Lab 26.7.1 或项目已经验证过的兼容版本；
- OrcaGym；
- OrcaPlayground；
- `orca` Conda 环境；
- `numpy`、`onnxruntime`、`Pillow`、`av` 等项目依赖；
- 可用的 `localhost:50051` 仿真服务。

#### 本 Demo 固定参数

本课程固定使用：

```text
assets/cae3c6559556dd4f/default_project/prefabs/g1_pick_usda
```

代码生成的 Actor 统一命名为：

```text
g1_navi
```

Asset、Actor 和 Layout 的基础概念已在入门文档中讲解，本篇只记录当前 Demo 使用的
固定资产路径和运行时 Actor 名称。

## 二、准备 Layout

### 2.1 新建场景

1. 启动 ORCA Lab；
2. 新建空 Layout；
3. 从资产中心拖入地面、桌子、货架、控制柜等环境资产；
4. 在 Outline 中给关键环境 Actor 使用容易辨认的名称；
5. 调整位置、旋转和碰撞关系；
6. 从多个视角确认路线中没有肉眼不可见的碰撞体；
7. 保存或导出 Layout JSON。

设计导航场景时，建议先从简单布局开始：

- 地面保持平整；
- 起点附近不要放太多物体；
- 路线宽度明显大于 G1 的身体宽度；
- 避免两个障碍物之间的缝隙过窄；
- 使用颜色明显的障碍物验证 RGB 检测；
- 在最终目标附近预留足够的停止空间。

本课程使用两个 Layout：

```text
/home/jason77/SY/demo1_v1.json   # 绿色桌子最小闭环
/home/jason77/SY/demo_v2.json    # 工厂巡检场景
```

### 2.2 环境和机器人分开加载

环境物体仍然由学员在 UI 中摆放，但本课程控制的 G1 不保存在 Layout 中，而是由
Python 通过 `OrcaGymScene.add_actor()` 和 `publish_scene()` 生成。

```text
手动 Layout：地面、桌子、货架、电气柜等环境
                         +
Python AddActor：本课程控制的 g1_navi
```

这样设计的主要原因不是普通关节控制，而是 Studio 相机流的注册方式。

## 三、为什么带相机的 G1 要通过 AddActor 加载

这里最容易混淆的是：**把 G1 手动放进 Layout 后，ORCA Lab 确实可以看到它，也可以在
界面中切换到它的第一视角。**这说明资产包含相机组件，而且 ORCA Lab 自己能够渲染该
相机，但不代表外部 Python 控制程序已经能够按 Actor 名称绑定并接收这路 RGB 数据。

这实际上是两条不同的相机使用链路：

```text
ORCA Lab 界面显示第一视角
    └── Studio 内部直接使用 Camera Component 渲染

Python 导航程序读取 RGB
    └── SetCameraSensorInfo 按 Actor 名查找相机
        └── CameraCaptureComponent 产帧
            └── WebSocket 7072 把 RGB 发送给控制程序
```

因此，手动放入 Layout 的 G1 可以被看见，也可以进行关节和运控控制；我们遇到的问题是：
在当前 Euler + Studio Camera API 链路中，Python 程序无法找到这个手动 Actor 对应的相机
实体，因而收不到导航所需的 RGB 帧。

官方 Euler Lesson 8 对这个现象给出了原因。Euler 环境后续使用 `LoadLocalEnv`，这条路径
不会填充 Studio 端用于相机查找的 `m_spawnedEntities`。因此，ORCA Lab 界面虽然能显示
第一视角，Python 调用 `SetCameraSensorInfo` 时仍可能得到 `Camera actor name not found`。

```text
手动把 G1 放入 Layout
        ├── ORCA Lab 显示第一视角：可以
        ├── Python 控制 G1 关节：可以匹配时可以
        └── Python 按 Actor 名绑定 RGB：当前链路失败
```

Lesson 8 因此先让机器人走 AddActor 路径：

```text
OrcaGymScene.add_actor
        ↓
OrcaGymScene.publish_scene
        ↓
Studio AddActor 创建 G1
        ↓
填充 m_spawnedEntities
        ↓
激活 CameraCaptureComponent
        ↓
EulerEnv LoadLocalEnv 导出 MJCF 并进行控制
```

经过 AddActor 后，Studio 会把机器人登记到 `m_spawnedEntities`，程序才能按 `g1_navi`
找到相机、激活 `CameraCaptureComponent`，并从 7072 获得连续 RGB 帧：

```text
Python AddActor 生成 G1
        ├── ORCA Lab 显示第一视角：可以
        ├── Python 控制 G1 关节：可以
        └── Python 按 Actor 名绑定 RGB：可以
```

所以我们选择 AddActor，并不是因为手动 G1 没有摄像头，也不是因为它完全不能控制，而是
因为本 Demo 的避障算法必须在 Python 中拿到每一帧 RGB。若只有 ORCA Lab 界面能看到画面，
程序却收不到像素数据，就无法计算绿色、黄色和深色区域占比，也无法让视觉信息改变运动
指令。

官方 Lesson 8 使用的是空关卡；本课程在此基础上做了一层工程适配：环境、桌子和电气柜
继续保留在手动打开的 Layout 中，Python 只发布机器人 Actor。核心代码等价于：

```python
scene = OrcaGymScene(grpc_addr=grpc_addr)
scene.add_actor(
    Actor(
        name="g1_navi",
        asset_path="assets/cae3c6559556dd4f/default_project/prefabs/g1_pick_usda",
        position=np.asarray((8.0, 0.463668, 0.0)),
        rotation=np.asarray((1.0, 0.0, 0.0, 0.0)),
        scale=1.0,
    )
)
scene.publish_scene()
```

也就是说，官方依据解释的是“为什么相机机器人要经过 AddActor”；“保留手动 Layout、
只发布 G1”则是本 Demo 在官方机制上做的组合方式。

官方依据：

- [OrcaPlayground Euler Lesson 8 文档](https://github.com/openverse-orca/OrcaPlayground/blob/main/examples/euler/08_video_capture/08_video_capture.md)
- [OrcaPlayground Euler Lesson 8 代码](https://github.com/openverse-orca/OrcaPlayground/blob/main/examples/euler/08_video_capture/video_capture.py)
- 本地对应文件：`examples/euler/08_video_capture/08_video_capture.md`

## 四、正确启动 ORCA Lab

打开 Layout 后点击运行，选择：

```text
No simulation program (manual launch)
```

不要同时启动 Empty Loop Simulation。否则两个程序可能同时推进同一仿真，造成画面在
正常站立与摔倒状态之间跳动。

正确工作流是：

```text
ORCA Lab 打开对应 Layout
        ↓
点击运行
        ↓
选择 No simulation program (manual launch)
        ↓
保持 ORCA Lab 运行
        ↓
在终端启动 Python Demo
```

## 五、三条通信链路

| 地址 | 用途 | 通俗理解 |
| --- | --- | --- |
| `localhost:50051` | OrcaGym 与 ORCA Lab 的 gRPC 控制 | Python 与仿真平台之间的“控制电话” |
| `localhost:7072` | `camera_head` RGB WebSocket | 相机持续发送图像的“视频频道” |
| `localhost:8765` | 可选浏览器调试页面 | 把 7072 图像和导航状态转给浏览器查看 |

ORCA Lab 可以直接显示 G1 第一视角，因此 8765 默认关闭。关闭浏览器窗口不会影响导航，
而 7072 图像流中断会让视觉导航进入等待或安全停止逻辑。

## 六、我们直接复用的底层运控

### 6.1 模型与配置

本课程不重新训练底层步态，直接复用官方 Euler Lesson 7 所使用的文件：

```text
assets/g1/models/dec_loco/model_6600.onnx
assets/g1/config/g1_29dof_hist.yaml
assets/g1/g1_29dof_camera.xml
```

官方 Lesson 7 将控制链路描述为：

```text
速度指令 + 本体状态 + 关节状态 + 历史观测
                    ↓
             ONNX 策略推理
                    ↓
       12 维 lower-body policy action
                    ↓
与 17 维 upper-body reference 合并
                    ↓
           29 维 q_target
                    ↓
tau = Kp × (q_target - q) + Kd × (0 - qd)
                    ↓
              关节执行器
```

官方依据：

- [OrcaPlayground Euler Lesson 7](https://github.com/openverse-orca/OrcaPlayground/blob/main/examples/euler/07_locomotion/07_locomotion.md)
- 本地对应文件：`examples/euler/07_locomotion/07_locomotion.md`

### 6.2 上层与下层的接口

导航层只输出：

```python
VelocityCommand(
    forward_mps=...,
    lateral_mps=...,
    yaw_rate_rps=...,
    walk_enabled=True,
)
```

`command_bridge.py` 把它送入 `G1Locomotion.set_commands()`。上层不直接修改 29 个关节，
因此调 waypoint 或 RGB 阈值时不需要重新训练步态。

### 6.3 如果想重新训练

官方 `examples/legged_gym` 提供足式机器人训练入口，支持 G1，并提供 SB3 PPO 与
RLlib APPO 两条链路。官方当前说明中，SB3 PPO 更适合先获得可用步态，RLlib APPO
仍属于需要继续调参的实验链路。

官方入口：

- [OrcaPlayground Legged Gym 使用指南](https://github.com/openverse-orca/OrcaPlayground/blob/main/examples/legged_gym/README.md)
- 本地 G1 配置：`envs/legged_gym/robot_config/g1_config.py`
- 统一入口：`examples/legged_gym/run_legged_rl.py`

训练命令结构为：

```bash
python examples/legged_gym/run_legged_rl.py \
  --config examples/legged_gym/configs/sb3_ppo_config.yaml \
  --train
```

训练前应按官方 README 把配置中的 `agent_name` 和场景机器人改成 G1，并核对观测、
动作、关节顺序和执行器类型。新训练出来的策略不保证能直接替换本课程的
`model_6600.onnx`：只有输入布局、动作含义、关节顺序、缩放和控制频率全部一致时，
才是可直接替换的 checkpoint。

## 七、29 个关节如何对齐 G1 Pick

### 7.1 数量关系

```text
29 个策略本体关节
├── 双腿 12：每条腿 6
├── 腰部 3
└── 双臂 14：每条手臂 7

g1_pick_usda 自身执行器
├── 本体执行器 29
│   ├── 位置执行器 15：双腿 12 + 腰部 3
│   └── 力矩 motor 14：双臂
└── 手部/夹爪辅助执行器 16：本课程保持零控制
```

场景中还可能有其他带执行器的机器人。它们属于整个 MuJoCo 模型的全局 `nu`，不能
误算成 G1 的夹爪数量。代码按 `g1_navi_` Actor 前缀区分 G1 与场景其他执行器。

### 7.2 名称对齐

`g1_pick_locomotion_env.py` 按 29 个关节全名寻找执行器，并要求每个关节恰好匹配一个：

```text
g1_navi_left_hip_pitch_joint
g1_navi_left_hip_roll_joint
...
g1_navi_right_wrist_yaw_joint
```

如果 Actor 名称不同，运行时关节前缀也会不同。因此 Actor 名、`agent_names` 和关节前缀
必须一致。这也是项目把机器人 Actor 统一命名为 `g1_navi` 的原因。

### 7.3 混合执行器转换

策略首先按训练模型计算期望 PD 力矩 `tau`。对于力矩 motor，可以直接发送 `tau`；
对于资产内置的位置执行器，需要反解它的控制目标：

```text
资产位置执行器近似关系：
tau = kp_asset × (ctrl - q) - kd_asset × qd

反解得到：
ctrl = q + (tau + kd_asset × qd) / kp_asset
```

这样同一个 29 关节策略才能同时驱动位置型腿/腰和力矩型双臂。实现位置：

```text
envs/euler/g1_vision_nav/g1_pick_locomotion_env.py
```

## 八、导航架构

### 8.1 世界坐标目标不是固定速度脚本

每次导航更新都会读取 G1 当前世界坐标和 base yaw，把世界坐标 waypoint 转换到机器人
坐标系：

```text
delta_world = waypoint_world - robot_world
delta_body  = body_rotation.T × delta_world
bearing     = atan2(delta_body_y, delta_body_x)
```

然后根据实时距离和 bearing 重新计算 `vx` 与 `yaw_rate`。所以机器人避障偏离主路线后，
控制器仍知道自己的当前位置，并能重新朝当前 waypoint 收敛。

### 8.2 RGB 视觉层做什么

当前 G1 头部相机俯角较大，主要用于近场安全。视觉层从 RGB 中检测当前固定场景内的
目标颜色，统计左、中、右区域占比：

```text
中央占比低：直接执行 waypoint 指令
中央出现障碍：降低前进速度，锁定较空一侧并增加侧移/yaw
画面连续清晰：逐步衰减视觉修正，重新交给 waypoint 跟随
```

RGB 颜色占比不是米制距离，也不是通用物体检测。本课程把它定位为“容易解释、容易
复现的视觉闭环基线”。未来可以替换为深度图、语义分割或学习型占用网络，但仍保留
相同的速度指令接口。

## 九、Demo A：绿色桌子最小闭环

对应分支：

```text
feature/demo1-v1-green-table
```

先切换到对应分支：

```bash
git switch feature/demo1-v1-green-table
```

运行入口：

```bash
python examples/euler/g1_vision_nav/run_green_table_crossing.py
```

固定条件：

```text
Layout: demo1_v1.json
G1 出生点: (4.7, -8.0)
绿色桌子: 约 (4.7, -5.0)
目标点: (4.7, 0.0)
```

起点到目标点的直线会穿过绿色桌子，因此它适合证明 RGB 是否真的改变了动作。

逻辑：

1. 点目标控制器实时计算前往 `(4.7, 0.0)` 的速度；
2. 7072 RGB 的中央区域出现足够多绿色像素；
3. 控制器选择画面较空的一侧；
4. 以小前进速度配合 yaw 绕开桌子；
5. 中央区域清空后，立即用当前位置重新计算目标指令；
6. 进入目标半径后停止。

这个 Demo 的教学价值是让学员先看懂最小闭环：

```text
目标跟随本来想直走
→ RGB 发现绿色桌子
→ RGB 临时改变速度指令
→ 清空后重新跟随目标
```

## 十、Demo B：工厂电气柜巡检

对应分支：

```text
demo1-v1-factory-navi
```

先切换到对应分支：

```bash
git switch demo1-v1-factory-navi
```

正式入口：

```bash
python examples/euler/g1_vision_nav/run_factory_inspection.py --route left
```

固定条件：

```text
Layout: demo_v2.json
G1 出生点: (8.0, 0.463668)
最终巡检点: (19.2, 4.3)

左线:
(9.0, 3.5) → (10.0, 3.5) → (13.0, 7.1) → (19.2, 4.3)
```

逻辑：

1. `WaypointRoute` 选择当前 waypoint；
2. `PointGoalNavigator` 根据实时位置和 yaw 生成主路径指令；
3. `FactoryColorObstacleDetector` 读取 7072 RGB；
4. 中央出现深色、绿色或黄色近场区域时，视觉层减速并修正侧移/yaw；
5. 连续清晰 10 帧后，修正量逐步降为零；
6. 中间 waypoint 进入 `0.40m` 到达圆即可切换；
7. 如果轻微越过中间点，但仍在 `0.40m` 路径走廊内，也允许进入下一段；
8. 最终巡检点不使用越过判定，必须进入 `0.20m` 半径；
9. 到达后停止，并把最后一帧保存到 `/tmp/g1_cabinet_left_rgb.png`。

左线已经在线验证并完成视频录制。右线命令为：

```bash
python examples/euler/g1_vision_nav/run_factory_inspection.py --route right
```

但右线目前属于后续开发，不应在课程中写成“已经验收通过”。可以把它设计成课后作业：
让学员调整 waypoint、障碍阈值和安全恢复参数，并提交路线视频与最终照片。

## 十一、核心代码阅读顺序

建议按数据流阅读，而不是按文件大小阅读：

1. `run_factory_inspection.py`：任务入口、路线、出生点和依赖组装；
2. `layout_scene.py`：AddActor 与相机注册；
3. `config.py`：端口、频率、速度、到达半径；
4. `waypoint_route.py`：当前路径点与切换条件；
5. `point_goal_navigator.py`：实时位姿如何变成速度；
6. `factory_inspection_navigator.py`：RGB 如何修正主路径；
7. `g1_camera_stream_env.py`：相机激活、取帧和最终照片；
8. `g1_vision_nav_env.py`：10 Hz 导航、50 Hz 运控与安全恢复；
9. `g1_pick_locomotion_env.py`：29 关节到混合执行器的适配；
10. `g1_factory_inspection_env.py`：任务级日志和验收指标。

## 十二、验收标准

### 12.1 绿色桌子 Demo

- G1 未跌倒；
- 7072 RGB 帧持续增长；
- 图像中实际观察到绿色桌面；
- 视觉层至少介入一次；
- G1 绕过桌子并进入目标半径。

### 12.2 工厂左线巡检

- G1 依次完成全部左线 waypoint；
- 运行中持续读取实时坐标和 yaw；
- RGB 风险实际改变过导航指令；
- 短暂碰撞可以恢复，连续确认跌倒必须停止；
- 最终距离不超过 `0.20m`；
- `/tmp/g1_cabinet_left_rgb.png` 存在且不是空白图；
- 保存俯视录屏、第一视角和终端最终结果。

## 十三、常见错误

### 13.1 相机 Actor 找不到

检查 G1 是否由当前脚本通过 AddActor 生成，以及 Actor 名是否为 `g1_navi`。不要把
Layout 中手动拖入的另一个 G1 与代码生成的 G1 混为同一个实例。

### 13.2 画面在站立和摔倒之间跳动

检查是否同时启动了 Empty Loop Simulation 和命令行任务。使用 manual launch，只保留
一个仿真推进程序。

### 13.3 关节匹配数量为零

查看报错列出的运行时关节前缀。Actor 名、`agent_names` 和关节前缀必须一致。

### 13.4 RGB 有画面但机器人不避障

先确认 frame index 持续增长，再观察中央颜色占比、阈值和 `latest_mode`。Studio 能显示
画面不等于 Python 一定连接了 7072 WebSocket。

### 13.5 G1 越过 waypoint 后不切换

检查当前点到达半径、过点走廊、路段起点以及实时坐标。最终点必须使用严格到达半径，
不能用中间点的“越过”规则代替。

## 十四、建议授课节奏

| 环节 | 建议时间 | 结果 |
| --- | ---: | --- |
| ORCA 架构与 Layout | 20 分钟 | 学员能解释三个软件层次 |
| 新建并保存 Layout | 25 分钟 | 得到环境 JSON |
| AddActor 与相机原理 | 20 分钟 | 能解释官方 Lesson 8 的限制 |
| 官方 G1 运控与关节适配 | 30 分钟 | 看懂 29→45 的适配边界 |
| 绿色桌子最小闭环 | 25 分钟 | 理解 RGB 如何改变指令 |
| 工厂左线巡检 | 35 分钟 | 到柜前并保存照片 |
| 日志、视频与复盘 | 15 分钟 | 完成验收记录 |

## 十五、课后扩展

1. 完成并在线验收工厂右线；
2. 把固定颜色检测替换为语义分割；
3. 接入真正的 depth 或点云，实现米制安全距离；
4. 增加多个巡检柜和任务顺序；
5. 为巡检照片增加目标检测或异常识别；
6. 按官方 Legged Gym 流程重新训练 G1 运控，并完成新的关节/观测适配；
7. 将 Layout、配置、视频、照片和 commit 固化为可复现实训发布包。
