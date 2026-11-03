# 手动调整布局、逐条采集、转换 LeRobot

任务：左臂将两只药瓶放进左侧紫框，右臂将两只药盒放进右侧蓝框。两臂交替搬运，只要求进入对应框内。

每次运行 `main`，程序读取当前保存的布局，自动计算抓取姿态和运动路径，执行并录制一条完整任务，检查成功与视频完整性后退出。你调整并保存下一种布局后，再运行同一命令。无需重写 IK，也无需手动给文本和视频配对。

## 1. 服务器准备

**数采服务器需要运行 OrcaLab 26.9.1 的画面渲染。** 仅模型训练才可以不启动 OrcaLab。建议把 OrcaLab 与采集 Python 放在同一台机器：相机接收器使用本机端口，单独修改 gRPC 地址不能实现完整跨机器数采。

- 安装/打开 OrcaLab 26.9.1，准备远程桌面和可用图形驱动。
- 将 `nursinghome / drag_classfication` 项目、当前 Layout 和对应资产迁到服务器。代码仓库不包含订阅资产，服务器必须能加载它们。
- 使用当前带夹爪的 G1，保持 `unitree_g1_1`、四件药品、两只框及三路相机名称。相机是 `head_cam`、`camera_left`、`camera_right`。
- 保存布局；运行时保持 OrcaLab 窗口可见，不要同时运行其他控制或录制程序。

先取得仓库：

```bash
git clone --branch handoff/medication-sorting \
  https://github.com/JasonZhw/OrcaPlayground.git
cd OrcaPlayground
```

后续更新使用 Git，不需要部署压缩包。在服务器的仓库目录中执行：

```bash
git switch handoff/medication-sorting
git pull --ff-only origin handoff/medication-sorting
```

本次改动当前仍在开发机工作区，提交并推送后，服务器才能通过上述命令取得新版。仓库不包含采集数据、订阅资产或虚拟环境，这些需要单独准备。

新版已合并旧模块。若服务器曾手动复制旧版代码，先检查 `git status` 并备份本地改动；Git 不会自动删除未跟踪的旧文件。旧入口 `run.py`、`scene.py`、`task.py`、`targets.py`、`kinematics.py`、`planner.py`、`environment.py`、`control.py`、`collect_manual.py` 不再使用。

已有可用 `orcalab_9_1` 环境直接复用。新服务器可按本机版本安装；软件镜像或官方包源需能提供对应版本：

```bash
conda create -n orcalab_9_1 python=3.12 -y
conda activate orcalab_9_1
python -m pip install -r examples/euler/medication_sorting/requirements-collection.txt
python -c "from orca_gym.environment.euler.orca_gym_euler_env import OrcaGymEulerEnv; from orca_gym.core.euler.orca_studio_bridge import OrcaStudioBridge; import av; print('Euler / bridge / video imports OK')"
```

版本号相同仍要核对导入：需要支持 Euler 与相机桥接 API 的 OrcaGym 包。若导入失败，应使用匹配的官方软件包/镜像，不要绕过这些接口。以上版本来自本机环境，尚未在你的新服务器安装验证。

第一次先检查场景，不驱动机器人：

```bash
python -m examples.euler.medication_sorting.main --mode inspect \
  --output "$HOME/medication_inspect_$(date +%Y%m%d_%H%M%S)"
```

检查 `scene_report.json` 是否包含四件药品和两只框，位置、尺寸是否符合当前布局。

## 2. 每次录制一条

保持在仓库根目录和 `orcalab_9_1` 环境，执行：

以下示例把数据放在 `$HOME/datasets`。云服务器请优先换成实际数据盘路径，例如 `/root/gpufree-data/datasets`，采集和转换命令中的路径要一起替换。

```bash
python -u -m examples.euler.medication_sorting.main \
  --dataset-root "$HOME/datasets/g1_med_sort_v2" \
  --execute
```

程序会：

1. 分配新编号，例如 `raw/train/episode_0000`，不会覆盖已有成功或失败条目。
2. 读取当前布局，**保留你摆放的位置**。不使用脚本基准 XY，不叠加额外位置/角度随机化。
3. 自动规划并执行四件药品抓放，记录头部、左腕、右腕三路完整画面。
4. 校验入框、状态动作、视频和帧索引，输出结果并退出。
5. 提示你检查视频、复位、微调药品并保存布局，再运行同一命令。

下一条之前，先停止/复位当前仿真，把四件药品放回桌面。保留机器人、桌子、框和相机的配置，先小幅改变一两件药品的 X/Y 或绕竖直轴的角度，避免初始穿插、悬空和互相遮挡，然后保存 Layout。不要把上一轮已经入框的状态直接当作下一轮起点。

原始 `episode.npz` 直接保存 `action.shape=(帧数,16)`、`state.shape=(帧数,30)`。动作第8项和第16项分别是左、右夹爪的闭合指令，范围0～1。底层连杆指令只在控制器内部展开，不进入动作数据。

默认轮换三种同义英文任务；也可以固定 `--task-index 0`、`1` 或 `2`。文本随 episode 保存，转换时自动写入 `task_index` 与任务表。

手动入口不更新旧 `collection_plan.json`。选用此流程后，不要在同一个数据目录重新运行旧计划式采集命令，以免计划中的编号与手动新增编号产生冲突。

当前运动规划并非任意位置都能成功。已有随机布局出现过关节轻微超限或抓持漂移；本入口保留检查，不掩盖失败。失败会返回非零退出码并保留 `result.json` 与原片；检查、复位、调整后再次运行会用新编号。

默认整个原始数据目录上限为 25 GB，启动前还要求至少 6.5 GB 空闲（4 GB 预留 + 预计单条 2.5 GB）。运行中每 5 秒检查容量、空闲空间和一小时超时；触及条件时中止并保留失败记录，等待子进程退出后才释放采集锁。检查间隔与收尾写盘可能产生少量超出，因此不要把预算设成磁盘的全部空闲量。

可以通过 `--max-gb`、`--reserve-gb`、`--episode-budget-gb` 调整预算。失败记录也占用容量，不会自动删除。

## 3. 留出验证布局

用没有参与训练的布局录制验证示范时，加 `--split val`：

```bash
python -u -m examples.euler.medication_sorting.main \
  --dataset-root "$HOME/datasets/g1_med_sort_v2" \
  --split val --execute
```

训练集、验证集按整条布局分开。不要把同一条示范复制到两个目录，也不要仅换一句任务文本就当作独立验证。程序保存 split，但不能替你判断两个手工布局是否足够不同。

## 4. 检查一条结果

以 `episode_0000` 为例：

```bash
cat "$HOME/datasets/g1_med_sort_v2/raw/train/episode_0000/result.json"
python -m examples.euler.medication_sorting.preview \
  --episode "$HOME/datasets/g1_med_sort_v2/raw/train/episode_0000/episode" \
  --output "$HOME/datasets/g1_med_sort_v2/raw/train/episode_0000/preview.mp4"
```

`result.success=true` 且 `episode/metadata.json` 中 `raw_valid=true` 才是转换候选。仍建议看完整视频，确认四件药品真实入框。`preview` 需要系统 `ffmpeg`。三路画面按原尺寸横向拼接，不裁剪、不缩放；640×480的三路画面加32像素标题栏后为1920×512，默认1倍速。原始训练视频仍是三份独立的640×480视频。

## 5. 安装独立转换环境

转换沿用已有 π0.5 工作流使用的 **LeRobot 数据格式 v2.1**，固定源码 revision，避免直接安装最新版改变目录结构。采集环境无需装 PyTorch。

在仓库根目录执行一次：

```bash
conda create -n med_lerobot python=3.12 -y
conda activate med_lerobot
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r examples/euler/medication_sorting/requirements-lerobot.txt
python -m pip install --no-deps \
  "lerobot @ git+https://github.com/huggingface/lerobot.git@0cf864870cf29f4738d3ade893e6fd13fbd7cdb5"
python -c "from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION; assert CODEBASE_VERSION == 'v2.1'; print(CODEBASE_VERSION)"
```

这里只安装数据转换所需依赖，再用 `--no-deps` 安装固定 LeRobot 源码，避免引入训练、硬件驱动等无关依赖。此环境不用于完整 LeRobot 训练。

还需系统 `ffmpeg` 提供 H.264 编码（例如 Ubuntu 上 `sudo apt-get install ffmpeg`）。转换使用 CPU，不需要 OrcaLab 运行，也不需要云端模型权重。

## 6. 转换训练集

先结束采集，再在 `med_lerobot` 环境运行。采集阶段已经直接生成每臂7个关节加1个夹爪量，共16维；本脚本只整理 LeRobot 文件格式、视频编码与元信息，不做动作维度转换。

```bash
python -u -m examples.euler.medication_sorting.lerobot_export \
  --source-root "$HOME/datasets/g1_med_sort_v2" \
  --output "$HOME/datasets/lerobot/local/g1_med_sort_v2" \
  --repo-id local/g1_med_sort_v2 \
  --split train
```

默认只选 `raw/train/` 下已经成功、完整的条目，列出并跳过失败条目；已标成功但发生文件损坏、相机索引错位、维度或配置不一致时会报错，不会静默带入。

转换保留：

| 内容 | 输出 |
|---|---|
| 三路完整画面 | `observation.images.head`、`left_wrist`、`right_wrist`；当前为 640×480，不裁剪、不缩放 |
| 状态 | `observation.state`：左臂7+左夹爪实测8+右臂7+右夹爪实测8，共30维 |
| 动作 | `action`：左臂7+左夹爪闭合程度1+右臂7+右夹爪闭合程度1，共16维；不做帧移位或差分 |
| 时间 | 50Hz，一行状态/动作对应三路视频的同一帧 |
| 任务 | 从原 episode 读取英文文本，自动关联 LeRobot 任务表 |
| 追溯信息 | `provenance/` 保存源路径、哈希、阶段和原始仿真时间/索引 |

原视频 FFV1 保留。训练视频重新编码为 H.264 CRF18，属于有损编码；保持完整构图和尺寸，不等于字节或像素完全相同。数值由 float64 转为 float32。

夹爪量是 0～1 的指令进度：0 张开，1 完成当前专家的闭合与夹持；不代表实测宽度。`provenance/export.json` 保存 `medication_closure_v1` 的映射版本和节点。推理必须使用同一套 `execution.Grippers.set(arm, closure)`，不能把 0～1 直接发给原来的 d。保留 30 维实测状态，不以闭合指令冒充状态。

本版本仅接受 `medication_sorting_raw_v2` 原生16维记录，旧18维记录不会转换。采集入口会检查目录中的协议和夹爪映射，拒绝混用；请使用新的 `g1_med_sort_v2` 目录。旧数据仍保留在原目录，不会自动删除。

转换会临时解码为 PNG，逐条编码后清理这些临时图片，建议至少留出数 GB 额外空间，不要在磁盘即将写满时转换。读回校验通过后才把 `.partial` 目录改成正式输出目录，并生成 `validation.json`。失败会保留 `.partial` 供排查，不覆盖它；修复后可选择新的输出名称重试。新增原始示范后也使用新输出目录重新转换，避免旧数据集被悄悄修改。

验证集单独转换，防止验证数据混进训练统计：

```bash
python -u -m examples.euler.medication_sorting.lerobot_export \
  --source-root "$HOME/datasets/g1_med_sort_v2" \
  --output "$HOME/datasets/lerobot/local/g1_med_sort_val_v2" \
  --repo-id local/g1_med_sort_val_v2 \
  --split val
```

`--split val` 兼容旧批量入口的 `raw/validation/` 与手动入口的 `raw/val/`。如果尚未采集验证示范，命令会报无可用数据，不生成空数据集。

重新检查已转换的数据：

```bash
python -m examples.euler.medication_sorting.lerobot_export \
  --source-root "$HOME/datasets/g1_med_sort_v2" \
  --output "$HOME/datasets/lerobot/local/g1_med_sort_v2" \
  --validate-only
```

校验包括数值逐行对比、任务映射、三路完整视频帧数/时间戳、关键阶段图像对比，以及官方 LeRobot reader 的动作 chunk 读取。**格式验证通过不代表抓取模型已经训练成功**：后续 ACT 或 π0.5 仍需适配本次双臂状态30/动作16维与三路相机，不能直接使用旧单臂训练配置。

当前演示、数据接口与使用边界见 [项目README](../../../README.md)，夹爪标定常量位于 `recording.py`。
