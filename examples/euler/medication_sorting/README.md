# 双臂药品分类：按采集流程阅读代码

场景为 OrcaLab 26.9.1 的 `nursinghome / drag_classfication`。左臂将两只药瓶放进左侧紫框，右臂将两只药盒放进右侧蓝框，交替搬运，只要求入框，不要求直立。

从采集数据到模型训练，请阅读[π0.5 LoRA 训练与推理说明](../../../docs/medication_pi05_training_inference.md)：包含当前完成情况、图像处理、双臂适配、开训条件和动作卡顿的验证方法，并附官方源码依据。

## 五个主文件

```text
main.py             启动、选择模式、自动编号、容量与进程管理
    ↓
workflow.py         读取场景 → 计算目标位置/姿态 → 编排抓取任务
    ↓
planning.py         IK求关节角 → RRT-Connect找路径 → 限速、限加速度采样
    ↓
execution.py        固定底盘 → 执行关节目标 → 保持夹爪指令
    ↔
recording.py        逐帧同步三路画面、状态、动作与英文任务
```

| 文件 | 输入 | 输出与主要修改点 |
|---|---|---|
| `main.py` | 采集命令、数据目录、模式 | 采集结果与新episode编号；默认每次录制一条当前布局 |
| `workflow.py` | OrcaLab导出的场景、任务对象 | 物体位置、抓取位置和姿态、任务阶段；原run/scene/targets/task集中于此 |
| `planning.py` | 当前关节状态、夹爪目标位姿、碰撞检查 | 关节路径及限速采样；包含IK与RRT-Connect，无CuRobo |
| `execution.py` | 关节路径、夹爪0～1闭合指令 | 实际物理运动与实测状态；保留到位、抓持漂移检查 |
| `recording.py` | 执行前状态/动作、同索引三路相机帧 | 原生16维动作、30维状态、视频、任务及完整性报告；同时定义数据协议与夹爪标定 |

阅读顺序：`main.main()` → `workflow.run_trial()` → `Trial.pick_place()` → `grasp_target()` → `planning.plan_motion()` → `execution.Executor.move()` / `tick()` → `recording.RawEpisodeRecorder`。

`workflow.py` 按“场景绑定与几何、任务配置与目标、任务阶段、单次运行”分区。抓取流程保持为接近、下探、闭合、抬升、横移、下放、释放、撤回。当前专家读取仿真物体真值，不通过视觉检测定位；摄像头为模型训练记录观察画面。

目标由现有物体的实际位姿和几何重新计算。改变XY或绕竖直轴的朝向后无需重写IK；资产型号、机器人朝向或尺寸大幅变化仍需校准和实测。规划通过不保证物理抓取成功。

## 一个启动入口

```bash
cd /home/jason77/SY/OrcaPlayground-medication-sorting
conda activate orcalab_9_1

# 默认模式：当前布局，自动规划并录制一条完整分类任务。
python -u -m examples.euler.medication_sorting.main \
  --dataset-root /home/jason77/SY/datasets/g1_med_sort_v2 --execute

# 仅检查场景，不控制机器人。输出目录必须不存在。
python -m examples.euler.medication_sorting.main --mode inspect \
  --output /home/jason77/SY/exports/med_inspect_01

# 当前布局只试跑动作，不录制。先复位场景并保存布局。
python -u -m examples.euler.medication_sorting.main --mode run --execute \
  --layout-mode current --output /home/jason77/SY/exports/med_test_01
```

采集后先检查结果和三路视频，再复位、微调药品并保存Layout，重新执行同一条采集命令。`--split val` 用于留出的验证布局。不要让两个控制进程同时操作仿真。

`--mode run` 下可以用 `--object calcium|melatonin|pharyngitis|herbbox|all` 选择单件或全部；加 `--record` 可录制调试示范。`current` 保留当前布局位置，`baseline` 使用本文件下述配置的基准位置；`--xy-jitter 0.005 --yaw-jitter-deg 3 --seed 100` 可添加可复现随机扰动。默认 `collect` 则始终采集当前布局四件物体，不接受这些试跑选项。

完整环境安装、逐条采集、预览、导出和服务器部署见 [手动工作流](MANUAL_WORKFLOW.md)。

## 看哪些输出

- 控制台 `GRASP_TARGET` 与 `grasp_targets.json`：每件药品实际抓取前计算的物体位置、夹爪目标位置、旋转矩阵与接近点，使用世界坐标、单位米；不是相机像素。后续目标尚未计算时不会提前出现。
- `scene_report.json` / `scene.xml`：实际对象绑定、几何尺寸及本次物理参数。
- `initial_layout.json`：本次初始位姿、布局模式及随机种子。
- `plans.json`：各阶段关节路径。
- `trace.json` / `final_qpos.npy`：动作执行诊断；state/action来自步进前，qpos_after来自步进后。
- `result.json`：执行结果、失败原因。
- `episode/`：录制的原始状态动作、三路视频、文本与索引。仅当 `success=true` 且 `raw_valid=true` 时作为成功候选。

## 常改参数

- 在 `workflow.py` 的 `BASELINE_XY` 修改基准摆放；当前布局模式不使用这些XY偏移。
- 在 `workflow.py` 的 `DROP_SLOTS` 修改框内XY分区，`RELEASE_BOTTOM_CLEARANCE` 修改药品底部距框底的高度（瓶8cm、盒5cm）；`placement_targets()` 按实际物体几何计算释放与悬停位置，输出到 `placement_targets.json`。
- 在 `workflow.grasp_target()` 查看抓取几何；`Trial.pick_place()` 查看闭合阶段与搬运顺序。
- 在 `execution.Executor.move()` 查看连续关节指令和到位检查；`planning.time_samples()` 用五次曲线限制速度与加速度；`ramp_gripper()` 平滑开合，释放用时1.2秒。
- 在 `recording.py` 的 `TASK_TEXTS` 查看三种英文同义任务。夹爪协议保留 `medication_closure_v1`，推理端必须使用一致标定。

每次导出XML时应用指尖 `condim=6`、全局 `impratio=2`，并保留已有闭链内部接触修正。执行时固定底盘，不修改订阅资产或虚拟环境包；物品由物理接触搬运，不焊接或传送物体。

## 数据与辅助工具

动作原生16维：左臂7＋左夹爪1＋右臂7＋右夹爪1。夹爪量为0～1的标定指令进度，不是实测宽度。状态30维保留两侧手臂及夹爪实际关节状态。三路画面为头部、左腕、右腕，保留640×480完整构图；按50Hz仿真时间同步，墙钟录制可能更慢。

| 辅助工具 | 作用 |
|---|---|
| `preview.py` | 成功示范按原尺寸横向拼接三视角MP4，默认1倍速；不缩放、不裁剪 |
| `lerobot_export.py` | 原生16维记录导出LeRobot v2.1，校验任务、视频、数值和官方reader；不做动作维度转换 |
| `batch.py` | 不同随机种子的小批量试验，顺序调用main，不并发控制 |
| `collect_dataset.py` | 预设计划的训练/验证批量采集；容量与数据协议检查也供main复用 |

手动采集与计划式采集不要混用同一数据目录。失败记录保留且占用容量，不会自动删除。训练/验证按整条布局分开；ACT/π0.5的双臂训练适配和异步模型推理仍需后续开发。

当前演示、布局和使用限制见 [项目README](../../../README.md)。
