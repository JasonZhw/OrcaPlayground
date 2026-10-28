# 手动工作流：执行、双相机采集、任务对齐与 LeRobot 转换

当前任务是把桌面上浅蓝色且最小的盒子放进左侧木托盘。机器人先放下左臂，再录制右臂抓取、搬运、松爪撤离和回位。只采集 `head_cam` 与 `camera_right`。

以下步骤从**已有与当前布局匹配的 `scene/plan.json`** 开始。换布局后的抓取重规划仍需先完成，不能仅修改下面的目录名或计划哈希。建议在同一个 Bash 终端依次执行；换终端后，需要重新设置批次、episode 和转换目录变量。

## 已采集数据

当前共保留 8 条用户验收示范、26,617 帧，来自 5 个布局批次；其中 7 条包含右臂回位。已按 [转换清单](lerobot_manifest_v1.json) 整理为 LeRobot v2.1，结果与云端训练包位置见 [数据转换说明](LEROBOT.md)。

以下启动命令对应最新一批布局，不是任意新布局的通用抓取计划。

## 最新批次

本批目录：`/home/jason77/SY/datasets/tray_transfer/layout_20261027_232508`。
已录制1条，3395组同步样本、67.90秒，包含右臂回位。抓姿相对竖直倾斜5°，重新求解当前布局的IK；抓取前闭合d=.97、对齐a=±.28/d=.90、搬运锁存d=1.45，释放高度12mm。

## 成功标准

本计划配置 `placement.acceptance="released_inside_tray"`。物体完整XY边界进入托盘（允许1mm接触容差）、底部接近托盘底面、夹爪松开后即可开始回位；不要求物体摆正、远离框边或通过静止速度/底面独立支撑检查。结束时确认右臂回到默认位置且盒子仍在框内。结果标准为 `released_inside_tray_and_return_home`。

旧抓取/搬运/放稳检查保留在结果的 `strict_quality_evaluation`，不与新任务标准混淆。之前 `xyshift_20261027_225137_retry04` 的58秒示范仍不含回位，原数据和验收记录保持不变。

## 当前布局启动命令

以下命令仅适用于本批导出的同一布局，输出目录必须不存在。调整任意物体后，重新读取场景并计算IK，不能直接复用旧坐标或只改哈希。

```bash
conda activate orcalab_9_1
cd ~/SY/OrcaPlayground
capture_root="$HOME/SY/datasets/tray_transfer/layout_20261027_232508"
capture_id=$(date +%Y%m%d_%H%M%S_%N)
episode_dir="$capture_root/episode_$capture_id"
python -m examples.euler.tray_transfer.scene_pick \
  --plan "$capture_root/scene/plan.json" \
  --output "$capture_root/diagnostics_$capture_id" \
  --online --contact-diagnostics \
  --record-episode "$episode_dir" \
  --task-index 1
```

先打开对应布局并进入运行场景；脚本从布局初始状态执行，执行期间不要再次点击Play或调整物体。每步等待两路相同仿真索引的画面，再执行动作。记录为50Hz仿真时间，墙钟录制通常更慢。

这一个命令同时执行专家轨迹和录制，不需要另起视频录制进程。`--output` 是诊断目录，`--record-episode` 才是训练原始记录目录。不要在录制命令里加入 `--preview` 或 `--development-preview`，参数校验会拒绝这种组合。再次采集前，将同一布局恢复为计划对应的初始状态，并生成新的 `capture_id` 与 `episode_dir`；不同示范可以选择不同的 `--task-index`。

原始采集在左臂准备阶段结束后开始。单独录制的桌面演示视频可以用于展示，但当前转换器需要下面的同步 PNG+NPZ，不能仅凭桌面录像生成对应动作数据。

## 保存内容

- `episode.npz`：15维实测state（右臂7+夹爪8）、9维绝对action（右臂7+a/d）、展开15维指令、锁存前请求、阶段、索引和时间。
- `frames/head`、`frames/right_wrist`：224×224 RGB PNG，按原宽高比缩放补黑边，无亮度增强。
- `metadata.json`：英文任务、状态/动作顺序、相机配置、判定与来源；`camera_frames.json`保存原生时间戳和索引。
- `--output` 指定的诊断目录：轨迹、执行源码、计划、模型信息与实际检查结果；与 episode 目录分别保存。

夹爪a/d为连续控制参数，展开为 `[0,0,a,d,0,0,-a,d]`，不是两个开关或牛顿单位的力。每帧为先观察后执行的 observation_t/action_t。

`--task-index 0/1/2`选择同一任务的三句英文表述，整条episode自动关联同一句，无需手动配对。

| 采集参数 | 写进 `metadata.json` 的完整英文文本 |
| --- | --- |
| `--task-index 0` | Pick up the smallest box on the table, which is light blue, and place it in the wooden tray on the left. |
| `--task-index 1` | Place the light blue box, the smallest one on the table, into the wooden tray on the left. |
| `--task-index 2` | Move the smallest box on the table, the light blue one, into the wooden tray on the left. |

这些文本定义于 [episode_recording.py](episode_recording.py) 的 `TASK_TEXTS`。它们描述同一个目标，不能把参数改成一条全新任务后，仍期望现有专家自动换物体或换目的地。专家目标由计划决定。

每个新布局先采1条。原始画面保持不变，亮度/对比度增强留到训练阶段；避免改变浅蓝色身份，不能代替位置、角度和干扰物布局的变化。后续扩充时按原布局划分训练/验证，再做增强；当前 8 条全部用于试训，尚无独立验证集。

原始 PNG+NPZ 和其中的 `training_ready=false` 保留不变；该标记描述原始记录，不代表转换没有完成。LeRobot 数据是单独生成的副本，不能用“格式通过”代替模型闭环成功率。返回 [代码框架](README.md)。

## 1. 采完后检查原始记录

先观察任务是否实际完成，再运行以下检查；只有目录存在不代表成功。如果程序中断或结果不通过，应查看同批诊断目录的 `result.json`，不要手动把失败标记改成成功。

在刚才的终端执行。检查已有记录时，先把 `episode_dir` 设置为那条记录的真实路径。

```bash
python - "$episode_dir" <<'PY'
import sys
from pathlib import Path
from examples.euler.tray_transfer.lerobot_export import load_episode

episode = Path(sys.argv[1]).resolve()
metadata, arrays = load_episode(episode)
print("Episode:", episode)
print("Task:", metadata["task"])
print("Success / complete:", metadata["success"], metadata["complete"])
print("Frames / fps:", metadata["frame_count"], metadata["fps"])
print("State / action:", arrays["state"].shape, arrays["action"].shape)
print("Includes return_home:", "return_home" in arrays["phase"])
print("Source checks passed; video roundtrip validation follows after conversion.")
PY
```

`load_episode()` 会检查成功/完整标记、英文任务、数值形状、动作展开、帧号、两相机索引、仿真时间间隔和 PNG 文件数量。它不是一次新的物理验收；图像解码和转换后的对齐由后续回读验证继续检查。

## 2. 明确选择哪些 episode 参与转换

转换器按清单的顺序分配新的 `episode_index`，不会自动扫描所有批次，也不会把失败或未完成的目录悄悄加入。

下面为刚才的单条成功记录创建一个新清单。需要多条时，在 `selected_episodes` 数组中显式增加那些已经验收的 episode 路径。每次转换前检查这份列表；只列新记录时，导出结果也只有新记录，不会自动合并原来的 8 条。

```bash
source_root="$HOME/SY/datasets/tray_transfer"
export_id=$(date +%Y%m%d_%H%M%S_%N)
manifest_path="$source_root/manifests/manual_$export_id.json"

selected_episodes=(
  "$episode_dir"
)

python - "$source_root" "$manifest_path" "${selected_episodes[@]}" <<'PY'
import json
import sys
from pathlib import Path
from examples.euler.tray_transfer.lerobot_export import load_episode

root = Path(sys.argv[1]).resolve()
manifest = Path(sys.argv[2])
entries = []
for value in sys.argv[3:]:
    episode = Path(value).resolve()
    relative = episode.relative_to(root)
    metadata, _ = load_episode(episode)
    entries.append({"path": relative.as_posix(), "layout_group": episode.parent.name,
                    "quality_notes": []})
    print(relative, metadata["frame_count"], metadata["task"])
if not entries or len({entry["path"] for entry in entries}) != len(entries):
    raise ValueError("Select at least one episode, with no duplicate paths")
manifest.parent.mkdir(parents=True, exist_ok=True)
with manifest.open("x", encoding="utf-8") as stream:
    json.dump({"description": "Manually selected accepted demonstrations", "episodes": entries},
              stream, ensure_ascii=False, indent=2)
    stream.write("\n")
print("Manifest:", manifest)
PY
```

已有 8 条数据的清单是 [lerobot_manifest_v1.json](lerobot_manifest_v1.json)，其中保留重复轨迹、未复位等质量说明。如需重新导出相同的 8 条，可以直接将 `manifest_path` 设置为该文件；如果合并新旧记录，保留旧清单中的这些说明。`layout_group` 仅是来源标签，当前转换器不会据此自动划分验证集。

## 3. 转换成 LeRobot v2.1

本机使用已经准备好的独立转换环境 `/home/jason77/SY/.venv-tray-lerobot`，无需升级仿真环境。以下路径按转换批次隔离，避免覆盖已有数据；内部数据集 ID 保持与当前 π0.5 配置一致。

```bash
data_home="$HOME/SY/datasets/lerobot_manual_$export_id"
dataset_root="$data_home/local/g1_lightblue_box_tray_v1"

/home/jason77/SY/.venv-tray-lerobot/bin/python \
  -m examples.euler.tray_transfer.lerobot_export \
  --manifest "$manifest_path" \
  --source-root "$source_root" \
  --output "$dataset_root" \
  --repo-id local/g1_lightblue_box_tray_v1
```

输出目录必须尚不存在。转换中断后不要把部分输出当作完成数据；排查失败后使用新的 `export_id` 和输出目录重试。此命令只在本地转换，不上传 Hugging Face。

产物结构如下，文件名以实际 `meta/info.json` 为准：

```text
dataset_root/
  meta/info.json
  meta/tasks.jsonl
  meta/episodes.jsonl
  data/chunk-000/episode_000000.parquet
  videos/chunk-000/observation.images.head/episode_000000.mp4
  videos/chunk-000/observation.images.right_wrist/episode_000000.mp4
  provenance/export.json
  provenance/episode_000000_metadata.json
  provenance/episode_000000_timeline.npz
```

两路视频从各自的有序 PNG 编码成 50 fps H.264。PNG 原件保持不变；视频是有损压缩，因此回读检查允许小幅像素差，不要求与 PNG 逐字节相等。

## 4. task、video、state、action 如何对齐

以一条 episode 的第 50 帧为例，所有位置均按从 0 开始的帧号计数：

| 数据 | 对应项 |
| --- | --- |
| 头部原图 | `frames/head/000050.png` |
| 右腕原图 | `frames/right_wrist/000050.png` |
| 状态 | `episode.npz` 中的 `state[50]`，实测 15 维 |
| 动作 | `episode.npz` 中的 `action[50]`，有效绝对目标 9 维 |
| 原生相机索引 | 两路都等于该行 `simulate_index`，当前记录器为 `1000050` |
| 转换后视频时间 | 两路视频均为 `50 / 50 = 1.0` 秒 |
| 任务 | 整条 episode 的同一句 `metadata.json → task` |

采集顺序是 `读取状态 → 拍摄当前画面 → 记录有效动作 → 执行动作`，即 `observation_t → action_t`。图像拍摄期间物理不推进，并且状态必须保持不变。每条 episode 的视频时间从 0 开始；原始 `sim_time` 可以从准备阶段之后的非零时刻开始。`timestamp_ns` 是墙钟采集时间，只作诊断，不用于训练视频对齐。

转换器向 LeRobot 的每一行同时写入两张图、该行 state/action 和完整英文 task。LeRobot 再将唯一文本登记到 `meta/tasks.jsonl`，在 Parquet 行内保存对应 `task_index`。采集命令中的 `--task-index 1` 是选择第二句文本；若它是本次导出遇到的第一种文本，LeRobot 可以将它编号为 0。因此检查**任务文本映射**，不要要求两个阶段的编号相同。

不需要手写字幕、按视频文件名配文本，或在视频剪辑软件中对齐。单独裁剪、变速或删除某路帧会破坏对应关系；如果以后要裁剪轨迹，需要同步处理图像、状态、动作和索引。

## 5. 回读验证后再交给训练

```bash
HF_HUB_OFFLINE=1 /home/jason77/SY/.venv-tray-lerobot/bin/python \
  -m examples.euler.tray_transfer.validate_lerobot \
  --root "$dataset_root" \
  --source-root "$source_root"
```

命令成功结束后，查看 `$dataset_root/validation.json` 中的 `passed: true`。验证内容包括：全部状态/动作值与 float32 转换后的原始值一致；所有视频的帧数和时间戳正确；关键图像与原件接近；每行任务文本正确；训练用的 50 步动作块在 episode 末尾以末帧动作补齐，不跨到下一条示范。

这里验证的是存储与对齐，不证明模型已经学会任务，也不会将失败的抓取变为成功示范。

## 6. 交给 π0.5 训练

上传完整 `dataset_root`，包含 `meta/`、`data/`、`videos/` 和 `provenance/`。训练包的 `--data-home` 应指向包含 `local/` 的父目录；如果仍使用本地这些路径，就是 `$data_home`，而不是 `$dataset_root`。云端迁移后相应替换为云端路径。

增加或更换训练数据后，使用新的 `--assets-dir` 重新运行 `norm`，再运行 `data-smoke` 和 `train`；不要沿用旧批次的归一化统计。具体云端命令见 [π0.5 训练说明](pi05_training/README.md)。保存本轮 manifest 和原始记录，便于将来追溯某条模型训练样本。
