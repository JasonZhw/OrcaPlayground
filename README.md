# 让大模型指挥机器人干活 · Agent-as-Policy

一个**不训练策略网络**的机器人操作框架：让视觉大模型决定「抓哪个、往哪动、什么时候放」，
由代码负责把所有决策变成精确、可执行、可校验的动作。

我们在 OrcaLab 仿真里用 Unitree G1 双臂机器人做了一个药品分拣任务作为验证场景：
左臂把两只药瓶放进左侧紫筐，右臂把两只药盒放进右侧蓝筐。

![演示预览](docs/assets/medication_sorting/preview.png)

## 为什么不用端到端 VLA

端到端策略网络能模仿示范，但一旦出错就停在那里：它不知道自己"为什么"失败，也没有第二次机会。

这套框架换了个思路——**大模型只做它擅长的语义判断，代码承担它擅长的精确执行**：

| 角色 | 负责什么 | 为什么 |
|---|---|---|
| 大模型 | 看图找物体、判断差多少、决定下一步动作、失败后换策略 | 语义泛化强，换物体不用重训 |
| 代码 | 像素→坐标、抓取点几何、逆运动学、路径规划、碰撞检查、到位校验 | 每一步都可验证，不会"说错就错" |

关键设计是**动作空间只有 8 个词**（`grasp / forward / backward / left / right / retreat / skip / aligned`），
模型永远不输出连续坐标——坐标由标定和几何算出来。

## 失败了怎么办

每个动作都返回结构化失败码（`ik_no_solution`、`lost_grasp`、`missed_bin` 等 13 种），
翻译成一句中文回注给模型，让它看着画面重新决定。同一类失败恢复成功后，会被记进经验库，
下次遇到优先复用。任务不会因为一次失败就终止。

## 能做什么

- **任务可配置**：一份 YAML 列出「谁、长什么样、进哪个筐、用哪只手」→ 逐件执行，代码不动
- **顺序由模型定**：开局观察时模型建议先做哪件，代码校验后采纳，不影响最终结果
- **换模型**：Anthropic Messages / DeepSeek / OpenAI 兼容端点，改配置即可
- **可视可复盘**：每次运行输出逐阶段事件、运动轨迹、预处理后的场景

## 快速开始

需要 OrcaLab 运行时在线（gRPC 50051 + 相机 7070/7072/7074），以及仓库根目录 `.env` 里的模型密钥。

```bash
conda activate orcalab
cd ~/SY/OrcaPlayground-medication-sorting

python examples/euler/medication_sorting/main.py --selfcheck   # 环境自检
python examples/euler/medication_sorting/main.py               # 跑完整任务
```

切换模型后端：

```bash
python examples/euler/medication_sorting/main.py --backend deepseek
```

完整安装步骤（OrcaLab 下载、资产订阅、conda 环境、依赖）见
[复现指南](examples/euler/medication_sorting/README.md#2-环境准备)。

## 代码结构

```
examples/euler/medication_sorting/
├── main.py      入口：模型后端、相机取帧、CLI
├── agent.py      主循环：逐物体 抓取 → 搬运 → 入框 + 失败恢复
├── tools.py        工具层：模型能调的每个动作（都返回 SkillResult）
├── failures.py          结构化失败码 / 恢复提示
├── lessons.py           经验库：失败码 → 成功恢复动作
├── scene.py             场景绑定、模型预处理
├── geometry.py          抓取点 / 放置点几何
├── planning.py          逆运动学（mink）、RRT-Connect 路径规划、限速采样
├── execution.py         仿真推进、夹爪控制、到位校验
├── grounding.py  单应矩阵视觉接地（像素 → 桌面坐标）
├── task.py          任务定义（读 tasks/*.yaml）
├── model_trim.py    模型裁剪（只保留双臂）
└── xml_strip.py    关节裁剪实现
```

`tasks/` 放任务定义，`layouts/` 放每个场景的单应矩阵标定。

## 文档

| 文档 | 内容 |
|---|---|
| [复现与使用](examples/euler/medication_sorting/README.md) | 环境安装、模型配置、运行、新场景接入、常见问题 |
| [技术报告](docs/medication_sorting_technical_report.md) | 架构、关键设计、实现细节、评测协议、能力边界 |

## 说明

- 目前**只在仿真验证**，未上真机。
- 能力边界（抓取高度手调、单应矩阵精度上限等）在技术报告与模块 README 里如实写明。
