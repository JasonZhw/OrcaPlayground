# G1 双臂药品分拣 · Agent-as-Policy

在 OrcaLab 26.9.1 中，左臂把两只药瓶放进左侧紫筐，右臂把两只药盒放进右侧蓝筐。
本项目不采用端到端 VLA，而是 **大模型做决策、代码做执行**：模型看图指出物体位置、给出下一步动作
（往前 / 夹 / 下放），代码负责把像素换算成桌面坐标、解逆运动学、规划路径并校验结果；
每个动作返回结构化失败码，失败会被翻译成自然语言反馈给模型重新规划。

开发分支：[`handoff/medication-sorting`](https://github.com/JasonZhw/OrcaPlayground/tree/handoff/medication-sorting)

![演示预览](docs/assets/medication_sorting/preview.png)

## 快速开始

需要 OrcaLab 运行时在线（gRPC 50051 + 相机 7070/7072/7074），以及仓库根目录 `.env` 中的模型密钥。
环境安装（OrcaLab 下载、场景资产订阅、conda 环境 `orcalab`、依赖）见
[模块 README 第 2 节](examples/euler/medication_sorting/README.md#2-环境准备)。

```bash
cd ~/SY/OrcaPlayground-medication-sorting
conda activate orcalab
export FABLE_API_KEY=sk-ant-...          # 或写进 .env

python examples/euler/medication_sorting/agent_policy.py --selfcheck    # 环境自检
python examples/euler/medication_sorting/agent_policy.py                # 跑完整任务
```

## 代码结构

```
examples/euler/medication_sorting/
├── agent_policy.py      入口：模型后端、相机取帧、main()
├── agentic_loop.py      主循环：逐物体 抓取 → 搬运 → 入框 + 失败恢复
├── fine_tools.py        工具层：模型能调的每个动作（都返回 SkillResult）
├── skills.py            结构化失败码 / CODE_HINT
├── lessons.py           经验库：失败码 → 成功恢复动作
├── workflow.py          场景绑定、抓取/放置几何、模型预处理
├── planning.py          IK（mink）、RRT-Connect 路径规划、限速采样
├── execution.py         仿真推进、夹爪控制、到位校验
├── visual_grounding.py  单应矩阵视觉接地（像素 → 桌面坐标）
├── arm_only_model.py    裁模型（只保留双臂）
└── mj_joint_strip.py    关节裁剪工具
```

使用与复现说明见 [模块 README](examples/euler/medication_sorting/README.md)；
技术设计、实现细节与评测协议见 [技术报告](docs/medication_sorting_technical_report.md)。

## 评测素材

`docs/assets/medication_sorting/` 保留了少量评测用图：

| 文件 | 内容 |
|---|---|
| `preview.png` | 三视角拼接的演示首帧 |
| `matched_camera_comparison.png` | 仿真与实拍相机画面对照 |
| `control_full_final_20261104.png` | 控制曲线最终版本 |
| `drag_classfication.json` | 原始 OrcaLab 布局 |
