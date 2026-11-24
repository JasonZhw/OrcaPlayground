# 公开流程验收

本页用于区分“材料与脚本齐全”“已有权重能完成任务”和“重新训练能复现成绩”。
目前还没有公开入口的新闭环分数。审计工具通过也不代表模型全部成功。
验证进度见 [REPRODUCTION_STATUS.md](REPRODUCTION_STATUS.md)。

## 第一阶段：已有权重闭环推理

先在空闲的 Linux NVIDIA 机器上按 README 安装仿真 profile。
不要把已有训练环境当作干净安装验证，也不要同时安装两个 Pillow profile。
所有命令均在本示例目录、`orca` 环境运行。

```bash
conda activate orca
python -m pip check
python reproduce.py check --profile simulation --strict-versions --cuda
```

严格检查对照 requirements 中的约束；缺包或版本不符返回非零状态。
Python 要求 3.12.x，精确补丁版本在环境报告中保留；原训练为 3.12.3。
cuRobo 的安装版本会检查，但提交身份尚不能由包版本证明，需保留固定提交安装日志。
这些检查不验证 tokenizer 访问权限、视觉资源内容或显存是否足够。
下载失败应先处理访问权限和网络问题，不能跳过模型与 tokenizer。

```bash
python reproduce.py download checkpoint --destination models/pi05-025000
python reproduce.py download tokenizer --destination models/tokenizer
python reproduce.py prepare-scene
python reproduce.py studio
```

按 [SCENE_AND_COLLECTION.md](SCENE_AND_COLLECTION.md) 同步四个视觉包。
在第二个同环境终端执行：

```bash
python reproduce.py studio --load
python reproduce.py check-scene --output outputs/scene-check
```

`check-scene` 会复位当前场景并抓取初始双相机图像；不要在其他采集或评测期间运行。
它使用相同运行目录锁及相机端口检查，不发送策略动作。
检查 `scene_check.json`、`head_cam.png` 和 `camera_right.png`：

- 两路应为 640×480 RGB、图像非空、无缺失材质和错误相机。
- 蓝色盒子、木框、机器人姿态与 V9 布局相符；腕部画面应对应右手。
- 记录 OrcaLab 中实际同步的资源版本或内容哈希。未知时填“未固定”。
- 与原采集录像的初始画面对照；当前包尚不提供经新入口验收的基准截图。

脚本不进行物体识别，非空画面不证明资源外观一致。
`configs/visual_resources.json` 明确保留未知内容版本，不以布局哈希代替纹理包哈希。

## 第二阶段：预处理对照与结果审计

在相同场景、权重、种子和保护参数下运行两组；不要覆盖或删除失败。
先完成一个配置的全部试验，再启动下一组。

```bash
python reproduce.py evaluate \
  --checkpoint models/pi05-025000 --tokenizer models/tokenizer \
  --output outputs/eval-pad-10 --trials 10 --seed 2026110700 --image-mode pad
python reproduce.py audit \
  --evaluation outputs/eval-pad-10 --output outputs/eval-pad-10-audit.json

python reproduce.py evaluate \
  --checkpoint models/pi05-025000 --tokenizer models/tokenizer \
  --output outputs/eval-legacy-10 --trials 10 --seed 2026110700 --image-mode legacy-stretch
python reproduce.py audit \
  --evaluation outputs/eval-legacy-10 --output outputs/eval-legacy-10-audit.json
```

`pad` 与训练转换的图像几何一致；`legacy-stretch` 对照历史缩放。
两种模式均使用当前保护器，因此均须报告自己的新成绩。
默认参数只是待验收配置，不表示已经选定最高成功率配置。
对照后选择配置，再用不同种子进行独立验收并冻结发布版本。

每组输出包含：

| 文件 | 含义 |
| --- | --- |
| `settings.json` | 请求次数、种子、预处理、关节跳变阈值及成功条件 |
| `provenance.json` | Git 提交/修改状态、源文件哈希、实际权重/tokenizer 哈希、环境与视觉资源版本状态 |
| `summary.json` | 全部试验及完成状态；中止和失败计入分母 |
| `trial_*/initial_*.png` | 每次试验的初始头部/右腕画面 |
| `trial_*/cameras.mp4` | 双相机录像 |
| `trial_*/trajectory.npz`、`predicted_chunks.npz` | 状态、命令、物理测量和原模型预测块 |
| `trial_*/result.json` | 单次结果与中止原因 |
| 独立目录外的 `*-audit.json` | 逐次记录检查、日志判据复核、证据文件 SHA256 |

审计拒绝未完成的试验组、额外/缺失的试验目录、视频帧数或频率错误、
与预测块不一致的命令，以及不满足日志中稳定条件的成功标记。
无动作/录像的早期错误保留为失败，并会导致证据审计不通过；不要删除后重算。
审计仅核对记录，不独立重演物理或判断视频中是否夹歪。
仍需人工查看全部录像，检查碰框、姿态、松爪和放置位置。

## 第三阶段：发布证据

将完整试验输出和独立审计报告存放到 Release 或专用 Hugging Face 仓库。
当前代码不会自动上传结果。发布前人工检查日志是否含个人路径或凭据。
不要把大量录像和模型文件提交进 Git。

公开报告至少记录以下信息：

- 代码提交、`project_dirty`、权重及 tokenizer 哈希。
- 环境、GPU、驱动、资源包内容版本；未知版本明确标出。
- 初始场景截图、预处理模式、种子、总请求数、完成数、成功/失败/中止数量。
- 审计结果、人工录像审核结果、完整证据下载位置。
- 实际安装、下载、推理耗时和峰值显存；尚未测量时不要填写最低硬件保证。

只有完整跑完并审核后，才更新 README 的“已验收配置与实测成功率”。
成绩只适用于所述场景与条件，不承诺不同机器或随机摆放达到 100%。

## 第四阶段：原始数据重新训练

先按 README 下载并完整转换公开 200 条数据。除了 200 条、277084 帧、
50Hz 和逐帧对齐，还应核对每条 state/action 与原始数据一致、
双相机顺序/补边图像样本，以及 `meta/stats.json` 与参考统计量的差异。
差异应逐项记录，不能只通过总帧数就认定数据相同。

随后用训练 profile 做短程启动检查，确认图像增强、归一化、全参数更新与有限 loss，
再按保存配方进行完整训练。正式计划保持 30000 步调度，
对第 25000 步检查点独立评测；不把下载的现成权重作为“本次重新训练”的结果。
完整训练尚未重新验证，原 GPU 为 96GB，batch32 的最低显存尚未测量。
新的训练结果使用同一闭环流程验收，与公开权重成绩分开报告。
