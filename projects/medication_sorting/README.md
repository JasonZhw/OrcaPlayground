# 药品分类：自动采集与π0.5

这是从独立实验目录整理的交接项目。固定底座G1使用右臂夹爪，把药盒放入紫筐、药瓶放入蓝筐；包含IK专家、RGB示范采集、LeRobot转换、π0.5训练、仿真推理与失败诊断。

**先阅读：[药品分类项目开发记录与复现手册](docs/medication_sorting/药品分类项目开发记录与复现手册.md)。**

## 当前结果与限制

截至2026-10-17，已有160条合格的单物品示范，每类40条，按128/32划分训练与验证；π0.5 LoRA已训练20,000步，检查点为 `19999`。

| 评估 | 结果 | 范围 |
| --- | --- | --- |
| 模型首次单物品推理 | 10/40 | 各物品独立重置 |
| 后续动作块对照 | 16/40 | 复用初态的调参结果，不是新的独立验收 |
| 连续专家参数筛选 | 0/75整轮 | 25组配置，配对初态 |
| 连续专家新初态复核 | 0/10整轮 | 不是VLA成绩 |

**四件连续分类尚未成功，不能把此项目当成已验收的分类系统。** 尚无新的严格连续四件RGB示范或基于它的模型。两药瓶为原尺寸70%，紫筐放大20%；摆放扰动仅为初始位置每轴±2cm、偏航±15度、保持直立。

## 目录

```text
medication_sorting/             本目录是独立项目根目录
├── tools/medication_sorting/   采集、转换、训练、推理、诊断及对应测试
├── tools/orca/environment.sh  项目环境辅助脚本
├── tests/                     适配器、数据契约与跨机测试
├── data/medication_sorting/scenes/
│                              原始与实验Layout，只含资产引用
├── data/medication_sorting/reports/
│   └── pi05-download-manifest.json  官方基座公开下载清单，不含权重
├── environment/medication_sorting/
│                              核心依赖参考、原机审计快照
├── docs/medication_sorting/    开发记录与完整复现步骤
├── docs/plans/                历史决策和后续方案
├── docs/reports/              尺寸、底座和路径诊断报告
└── docs/发布范围与迁移说明.md  GitHub与单独传输文件的边界
```

本目录之外的仓库文件不属于本交接项目。不要在仓库根目录安装本项目的依赖或运行其测试。当前实现使用既有OrcaGymLocal，不是仓库中新建的Euler示例，也未接入CuRobo。

## 在新电脑开始

已验证的原机为Ubuntu24.04、Python3.12、RTX5090约32GB显存，OrcaLab/OrcaGym均为26.9.1。其他GPU、Windows或WSL尚未完成端到端验证。

在此README所在目录打开终端。若仓库克隆在 `~/OrcaPlayground`：

```bash
cd ~/OrcaPlayground/projects/medication_sorting
export SY_ROOT="$PWD"
```

之后按复现手册第4至第6章依次安装环境、取得资产权限、加载场景与模型。注意：

- `.venv`、`software/openpi` 和 `software/OrcaPlayground` 均在**本目录内**重新建立；后两者按手册固定commit，不直接复用外层仓库的main版本。
- 场景JSON不包含资产。朋友需用自己的账号取得 `NursingHome`、`unitree_g1_gripper`、`AutoProductionLine_Warehouse`、`Medical` 中相应资产的访问权限。
- 微调模型、数据、tokenizer及官方基座未上传GitHub；按发布范围说明单独获取。没有模型文件时不能直接运行已有策略推理。
- 用已有模型推理不需要再次训练。继续优化时先阅读尺寸和连续路径诊断报告，不直接重跑整个160条流水线覆盖旧输出。
- 截图使用当前用户 `~/.orcalab`，特殊安装可设置 `ORCALAB_CAPTURE_DIR`；OrcaLab和采集进程应使用同一用户、同一HOME。

## 关键入口

| 任务 | 入口 |
| --- | --- |
| 备份当前场景并加载指定Layout | `tools/medication_sorting/prepare_scene.py` |
| 单物品采集与完整回合验收 | `collect_episodes.py`、`verify_episodes.py` |
| 四类均衡采集与160条流水线 | `collect_balanced.py`、`run_randomized_experiment.py` |
| 训练环境、转换及π0.5适配 | `setup_pi05.py`、`pi05_python.sh`、`convert_episodes.py`、`pi05_adapter.py` |
| 已有模型推理及单物品评估 | `infer_policy.py`、`evaluate_policy.py` |
| 严格连续专家测试 | `continuous_expert.py` |
| 尺寸、IK和接触诊断 | `geometry_audit.py`、`geometry_probes.py`、`physics_diagnostics.py` |

入口表中的短文件名都位于 `tools/medication_sorting/`。完整参数以手册和各入口 `--help` 为准。`night_expert_search.py` 保留了历史有期限实验的调度方式，不会在克隆后自动启动。

## 离线回归

先按手册安装仿真依赖并取得固定版本OpenPI源码；不需要模型权重和真实资产：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests tools/medication_sorting/tests
```

未安装实际训练依赖层时，对该依赖层的集成测试会跳过。离线测试通过不代表新电脑已通过GPU训练、相机或真实抓放验收。
