# 药品分拣 · Agent-as-Policy

Unitree G1 双臂机器人在 OrcaLab 里把药瓶放进紫色筐、药盒放进蓝色筐。

**一句话**：大模型只看画面做判断，代码负责算坐标、解逆运动学、执行和校验。

---

## Agent-as-Policy 是什么

视觉动作模型需要大量示教数据，换个物体就得重采，失败时也说不清原因；纯脚本把坐标写死，
物体一偏或抓滑一次，整条流程就停。所以我们走的路线：**模型只做判断，代码负责把判断变成动作**。

具体做法：

- **位置由模型报，坐标由标定换算。** 模型在俯视图上指出"物体中心在这个像素"，程序用事先标定好的
  单应矩阵把它换成桌面坐标（残差约 7.5 mm）。抓取和放置只给语义动作——往前一点、可以夹了、可以下放——不带坐标。
- **动作只有八种。** grasp / forward / backward / left / right / retreat / aligned / skip，
  没有连续参数，所以即使判断偏一点，也只是多走一小步，不会把机械臂送进碰撞。
- **失败会被告知。** 每个动作返回结果和失败原因（IK 无解、路径被挡、没夹住、没进筐……），
  翻译成中文交回模型换做法；同类失败下哪种处理奏效过会被记下来复用（`lessons.py`）。
- **几何做最后确认。** 物体是否真的进了筐由几何判定，模型偶尔看错左右时按几何纠偏。

![闭环图](../../../docs/assets/medication_sorting/agent_loop.png)

---

## 1. 任务

对桌面上的药品进行分拣：两只药瓶放进左侧紫色筐，两个药盒放进右侧蓝色筐。

左臂负责药瓶（侧握，水平前伸去夹），右臂负责药盒（顶握，从上方竖直下压）；
只要求放进对应的筐，不要求摆放姿态。

![任务场景](../../../docs/assets/medication_sorting/task_scene.png)

---

## 2. 环境准备

### 2.1 需要什么

| 项 | 要求 |
|---|---|
| OrcaLab | ORCA Lab 开发者版（免费），需要登录后订阅 **4 个资产包**（见 2.2 ②） |
| Python | 3.12|
| GPU | 按松应官网的系统要求配置（见下方链接），主要供 OrcaLab 渲染；本项目的物理仿真跑 CPU，模型走 API |

官方入口：

- 下载安装包：<https://www.orca3d.cn/download/>
- 开发文档（含系统要求、资产订阅）：<https://docs.orca3d.cn/>
- OrcaGym 说明：<https://github.com/openverse-orca/OrcaGym>

### 2.2 安装步骤

**① 装 OrcaLab**

到 <https://www.orca3d.cn/download/> 下载 ORCA Lab 安装包并安装（Windows / Linux 都有），
按官网系统要求确认显卡与显存。装完后打开一次，确认能加载场景。

**② 订阅资产（共 4 个包，缺一个都加载不完整）**

在 OrcaLab 里登录账号，按下面的名称逐个订阅并下载：

| 资产包 | 提供什么 |
|---|---|
| NursingHome | 房间、桌子、床、窗帘、轮椅、试管架等场景物件 |
| unitree_g1_gripper | 机器人本体 `unitree_g1_1`（含夹爪） |
| AutoProductionLine_Warehouse | 紫色筐 `Box_Purple`、蓝色筐 `Box_Blue` |
| Medical | 4 件药品：`Medical_32 / 31 / 23 / 12` |

仓库里的 [`drag_classfication.json`](../../../docs/assets/medication_sorting/drag_classfication.json)
就是本项目的布局（自己配置的，引用了上面 4 个包）。在 OrcaLab 里导入这个布局，
即可得到与本 README 截图一致的场景：机器人站在桌前、左右两个筐、四件药品。

**③ 建 conda 环境**

```bash
conda create -n orcalab python=3.12 -y
conda activate orcalab
```

**④ 装 Python 依赖**

```bash
cd ~/SY/OrcaPlayground-medication-sorting
pip install -r examples/euler/medication_sorting/requirements.txt
```

`requirements.txt` 内容：

```
mujoco==3.12.0
numpy==2.2.6
scipy==1.16.2
grpcio==1.66.1
protobuf==5.29.6
av==19.0.2
Pillow==12.0.0
mink==1.4.0          
```

**⑤ 自检**

```bash
python examples/euler/medication_sorting/agent_policy.py --selfcheck
```

### 2.3 启动 OrcaLab 并确认端口

在 OrcaLab 里打开场景、确认机器人就位后，终端检查 4 个端口：

```bash
ss -tln | grep -E '50051|7070|7072|7074'
# 50051 = gRPC 控制   7070 = 头部相机   7074 = 左腕相机   7072 = 右腕相机
```

场景需要重置时：`pkill -f orcalab.main` 后重新打开。

---

## 3. 配置模型 API

模型密钥**不写进代码**，放在仓库根目录的 `.env`（已被 `.gitignore` 排除，不会提交）。

```bash
cd ~/SY/OrcaPlayground-medication-sorting
cat > .env <<'EOF'
FABLE_API_KEY=sk-ant-xxxxxxxx
FABLE_MODEL=claude-fable-5-2
FABLE_BASE_URL=https://api.anthropic.com
EOF
chmod 600 .env
```

程序启动时会自动向上查找并加载 `.env`（`agent_policy._load_dotenv`），已存在的环境变量不会被覆盖。

### 换模型供应商

三个变量都可由环境变量覆盖，所以换成任何 **Anthropic 兼容端点**只需改值。例如 DeepSeek：

```bash
FABLE_BASE_URL=https://api.deepseek.com/anthropic
FABLE_API_KEY=<DeepSeek 的 key>
FABLE_MODEL=deepseek-flash
```

> ⚠️ 必须选**带视觉**的模型（`deepseek-flash`）。`deepseek-v4-pro` 不处理图片，
> 所有看图判断会退化成纯文本，任务一定失败。

也可以用 OpenAI 兼容端点（DeepSeek / Qwen-VL / GPT-4o）：

```bash
python agent_policy.py --backend openai
# 读取 OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL
```

---

## 4. 运行

```bash
cd ~/SY/OrcaPlayground-medication-sorting
conda activate orcalab

# ① 环境自检：依赖 / 端口 / 相机 / 模型连通性
python examples/euler/medication_sorting/agent_policy.py --selfcheck

# ② 跑完整任务（4 件）
python examples/euler/medication_sorting/agent_policy.py
```

参数：

| 参数 | 说明 |
|---|---|
| `--selfcheck` | 不动机器人，只检查环境 |
| `--output <dir>` | 输出根目录（默认 `./agent_out`） |
| `--backend anthropic\|openai` | 模型后端 |
| `--no-capture` | 关闭相机（调试用，视觉工具会全部失效） |
| `--full-body` | 用完整机器人模型（调试用，默认只用双臂） |

### 输出在哪里

每次运行新建一个目录，不会覆盖历史：

```
agent_out/20261118T101500Z_ab12cd34/
├── episode_result.json      每个阶段的记录（status / events / 失败码）
├── plans.json               每次运动的关节路径
├── motion_context.json      最近一次运动的目标与约束
├── prepared.xml             OrcaLab 导出的原始模型
└── prepared_armonly.xml     裁剪后的双臂模型
```

成功判据：`episode_result.json` 里 `"status": "complete"`。

---

## 5. 一次执行长什么样

```
🔭 观察场景 → 决定任务                    模型看全局俯视图，说看到什么、打算怎么做
🎯 物体 → 目标筐
   [工具] locate(...)                    模型报物体像素 → 单应矩阵换算成 (x, y)
   [推理] 物体中心 / 握法 / 抓取点怎么推出来
   [工具] move(approach_xy / approach / descend)
   [推理] + [执行] + [工具]               逐轮判断方向，直到 grasp
   [工具] grasp_preset                   三段式闭合 → 接触确认
   [工具] move(test_lift)                抬 3cm 验证是否夹稳
   [工具] plan_placement                 ｜模型决定筐内落点
   [工具] move(lift_safe / carry)        ｜抬到筐口上方再水平搬
   [工具] move(align_place …)            ｜物体进筐口范围才放行
   [工具] release → 等落底 → 校验
   [工具] move(home)
```

失败时不会停：失败码 → 中文提示 → 喂回模型 → 重新判断（重抓最多 3 轮、搬运绕行最多 6 次、微调失败改判）。

---

## 6. 代码结构

| 文件 | 职责 |
|---|---|
| `agent_policy.py` | 入口：模型后端（Anthropic / OpenAI 兼容）、相机取帧、`main()` |
| `agentic_loop.py` | 主循环：逐物体「抓取 → 搬运 → 入框」+ 失败恢复 |
| `fine_tools.py` | 工具层：模型能调的每个动作，全部返回 `SkillResult` |
| `skills.py` | 结构化失败码、`CODE_HINT`、`classify()` |
| `lessons.py` | 经验库：失败码 → 成功恢复动作（跨轮累积） |
| `workflow.py` | 场景绑定、抓取点/放置点几何、模型预处理 |
| `planning.py` | mink 逆运动学、RRT-Connect 路径规划、限速采样 |
| `execution.py` | 仿真推进、夹爪控制、到位校验 |
| `visual_grounding.py` | 单应矩阵视觉接地（像素 → 桌面坐标） |
| `arm_only_model.py` | 裁模型：只保留双臂 + 夹爪 |
| `mj_joint_strip.py` | 关节裁剪的底层实现 |

---

## 7. 从零复现

1. 取代码：`git clone https://github.com/JasonZhw/OrcaPlayground.git`，切到 `handoff/medication-sorting`
2. 装 OrcaLab，按第 2.2 节 ② 订阅 4 个资产包，并导入 `drag_classfication.json` 布局
3. 建 `orcalab` 环境、装依赖（第 2.2 节 ③④）
4. 配模型密钥（第 3 节）
5. 打开该布局，确认机器人站在桌前、两个筐和四件药品就位
6. `--selfcheck` 通过后运行（第 4 节）

`layouts/homography.json` 已随仓库提供。换场景、换机器人或挪动桌子后必须重新标定，
否则 `locate` 会把像素映射到错误位置。

### 重新标定 `homography.json`

`HomographyGrounder` 用一张 3×3 矩阵把头部相机的像素映射到桌面坐标。标定方法：
把参考物体依次移到桌面网格点，让模型报每个点的像素，再用 RANSAC 拟合（残差约 7 mm）。

```python
# 需要 OrcaLab 在线；一次标定即可，结果写进 layouts/homography.json
import asyncio, grpc
from orca_gym.core.euler.orca_studio_bridge import OrcaStudioBridge
from orca_gym.protos.mjc_message_pb2_grpc import GrpcServiceStub
from examples.euler.medication_sorting.agent_policy import AnthropicBackend
from examples.euler.medication_sorting.execution import SortingEnv
from examples.euler.medication_sorting.workflow import Scene
from examples.euler.medication_sorting.visual_grounding import calibrate

loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
ch = grpc.aio.insecure_channel("127.0.0.1:50051")
bridge = OrcaStudioBridge(GrpcServiceStub(ch))
xml = loop.run_until_complete(bridge.load_model_xml())
loop.run_until_complete(bridge.pause_simulation())
print(calibrate(bridge, Scene(xml), SortingEnv(xml), "left", AnthropicBackend(), loop=loop))
```

## 8. 常见问题

| 现象 | 处理 |
|---|---|
| `--selfcheck` 报端口未监听 | OrcaLab 没起 / 场景没打开 |
| 自检报"没抓到任何帧" | 相机流没开，重启 OrcaLab；或 `--no-capture` 下视觉工具必然失效 |
| `locate` 连续 3 次失败 | 检查 `layouts/homography.json` 是否匹配当前场景；或模型没返回像素 |
| 反复 `ik_no_solution` | 物体太远/太偏；可调 `workflow.grasp_target` 里的抓取高度与偏移 |
| `[执行] 输出无法解析（unknown）` | 模型输出格式跑偏，程序会自动重问；连续 6 次才失败 |
| 任务跑完但某件没进筐 | 看 `episode_result.json` 的 `status` 与最后几条事件 |

---

完整的技术设计、实现细节、评测协议与局限，见
[技术报告](../../../docs/medication_sorting_technical_report.md)（同目录下有 `.docx` 版本）。

---

## 9. 参考框架


| 参考 | 借鉴了什么 |
|---|---|
| [SayCan](https://say-can.github.io/) | 大模型输出**语义动作**，底层由可执行 skill 完成；模型不碰底层控制 |
| [ReAct](https://arxiv.org/abs/2210.03629) | 推理与行动交替：`[推理] → [执行] → [工具] → 观察结果` |
| [Code as Policies](https://code-as-policies.github.io/) | 每个能力是**带返回值的函数**，模型只负责挑函数和参数 |
| [Voyager](https://voyager.minedojo.org/) | 把失败经验沉淀下来跨轮复用（对应本项目的 `lessons.py`） |
| [RobotUse](https://arxiv.org/abs/2610.04929) | backend 承担几何与运动、视觉**点目标**接地、持久化经验、`place ≠ release` |

