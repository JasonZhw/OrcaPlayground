# 版本与运行条件

## 原训练环境

| 软件 | 版本 |
| --- | --- |
| Python | 3.12.3 |
| LeRobot | 0.6.1，PyPI，`lerobot[pi,training]` |
| torch / torchvision | 2.8.0+cu128 / 0.23.0+cu128 |
| transformers / accelerate | 5.5.4 / 1.16.0 |
| datasets / huggingface_hub | 4.8.5 / 1.33.0 |
| torchcodec / tokenizers | 0.7.0 / 0.22.2 |
| NumPy / draccus | 2.2.6 / 0.11.6 |
| CUDA 运行时 | 12.8，PyTorch wheel 自带 |
| 原系统 | Linux 6.6，glibc 2.39 |
| 原硬件 | 单 RTX PRO 6000 Blackwell 96GB，24 CPU 核，177GB RAM |

`reference/training/requirements.txt` 是训练记录里的完整包列表，
`reference/training/pyproject.toml` 是原 uv 项目，均保持原字节并有哈希。
便携 `requirements.txt` 只锁关键运行包，不是所有平台的传递依赖锁文件。
原训练Pillow为12.3.0；OrcaLab26.9.1要求Pillow12.0.0。
`requirements-training.txt` 与 `requirements-simulation.txt` 分别选择这两个版本，
两种profile应使用独立机器/环境，不能混装后声称原训练环境不变。
保存的配置、原始材料和固定 Hub revision 是版本依据，不根据浮动 main 推断版本。

## 仿真与采集环境

OrcaLab/OrcaGym 26.9.1，MuJoCo 3.12.0，SciPy 1.16.2。
cuRobo 固定 `78fd485fa82d9b9a063fb4985e371814587e666a`，
使用 `MotionPlanner` / `MotionPlannerCfg`，不是旧的 `MotionGen` 示例 API。
需要 CUDA 编译工具链；相机依赖 OrcaStudio 图形引擎及资源订阅。
FFmpeg/ffprobe 用于原录像的解码、采集录像编码与帧数校验。

复现检查在名为 `orca` 的克隆仿真环境中执行，Python 3.12.14。
它验证便携场景与接口，不等于证明原 Python3.12.3 训练环境已从零装好。
克隆环境原有 `jsonschema-path` 与两项 OpenAPI 包的版本冲突，
仿真profile明确约束为 `>=0.4.5,<0.5.0`；
目标机器仍需通过 `pip check`，不能静默忽略其他依赖冲突。

OpenPI 图像工具来自 `215abfb217dbac7d5f1273282331b9b1866c0479`，
文件原样收录并带 Apache-2.0 许可证，直接依赖 NumPy/Pillow。
没有安装整个 openpi-client，因此无需通过 `--no-deps` 绕过它的 NumPy 上限。
原转换器所用 openpi-client 的精确提交尚无证据，不能承诺像素级历史等价。

## 数据契约

200条、277084帧、50Hz、1个任务、LeRobot v3.0、图像内嵌Parquet。
头部键 `observation.images.image`，右腕键 `observation.images.wrist_image`。
原录像1280×480，左右各640×480 RGB，补黑边到224×224，
有效图像224×168，上下各28像素黑边。

状态和动作顺序为右臂 shoulder_pitch、shoulder_roll、shoulder_yaw、
elbow、wrist_roll、wrist_pitch、wrist_yaw，随后内外两根夹爪主关节。
关节角为绝对弧度；夹爪命令0.5张开、-0.7闭合，实测状态与命令值的范围不同。
状态动作使用MEAN_STD，不能替换成QUANTILES或增量动作。
完整精度统计量见 `reference/training/stats.json`。

## 检查点与续训

公开025000检查点包含模型和预处理/后处理器，共7份推理文件。
模型9354050752字节，SHA256：

```text
65ea04941cf5f0a6de906baa58e3d39b24b5aaa8cc7f6974b234912b8e2ea78c
```

公开文件未包含完整优化器、调度器及随机数状态。
从这些权重重新开始优化不等于原运行的无缝续训；
需要无缝续训时应保留 LeRobot 完整本地训练检查点。

运行前可用 `python scripts/collect_versions.py --help` 查看版本导出参数。
升级 LeRobot、OrcaLab、cuRobo 或转换逻辑后，应作为新实验重新记录版本和评测。
