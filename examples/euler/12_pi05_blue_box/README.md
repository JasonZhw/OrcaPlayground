# V9 蓝色物品夹取：200 条示范与 pi0.5 全参数微调

任务：G1 固定底座，右臂夹起蓝色 Intel 盒子，搬运到木框内并主动松爪。
本示例提供原始数据下载、cuRobo 自动采集、Euler 场景、数据转换、完整训练配置、
权重下载与闭环录像评测。训练使用 LeRobot 的 pi0.5 实现，全参数更新。

## 先选择学习路线

| 目标 | 需要完成 | 可以跳过 |
| --- | --- | --- |
| 快速复现夹取任务 | 安装仿真环境、同步场景资源、下载完整策略包和 tokenizer、闭环推理与结果检查 | 示范数据下载、采集、数据转换、训练 |
| 用课程数据重新训练 | 下载公开的 200 条示范、转换数据、准备独立训练环境、全参数微调、评测新检查点 | 重新采集 |
| 自行采集与训练 | 配置任务与场景、cuRobo 采集、转换新数据、全参数微调、评测新检查点 | 课程原始数据下载 |

**第一次操作，先走快速复现路线。**公开策略包已经使用 200 条示范完成训练，
下载的是第 25,000 个训练步的完整模型及处理器，不是 25,000 条数据，也不是 25,000 个训练轮次。
公开 200 条示范是可选的训练材料，只使用现成模型推理时无需下载。
飞书教程对应第三、四、五、九章；第六至八章用于学习采集、转换与训练。
所有命令中的路径均相对于本示例目录。
验证范围见 [复现状态](docs/REPRODUCTION_STATUS.md)。
安装、初始画面检查与结果核验步骤见
[公开流程验收](docs/ACCEPTANCE.md)。

## 资源与版本

| 资源 | 链接 | 固定 revision |
| --- | --- | --- |
| 原始 200 条数据 | [数据集](https://huggingface.co/datasets/xiaojia666/g1-blue-intel-single-target-200) | `3045d12264211a0d239b2abb3553d2d243df7c99` |
| 25000 步全参数权重 | [模型](https://huggingface.co/Harrysunshine/g1-pi05-v9-single-target-200) | `70ca815cb53de79ae867f83be0bdac3fbc4d7498` |
| 初始 pi0.5 模型 | [LeRobot pi05 base](https://huggingface.co/lerobot/pi05_base) | `22c30e5b34f9e7ab738b1b34fc2fc5ecfe2d28f0` |
| tokenizer | [PaliGemma](https://huggingface.co/google/paligemma-3b-pt-224) | `35e4f46485b4d07967e7e9935bc3786aad50687c` |
| 官方 OpenPI 仓库 | [OpenPI](https://github.com/Physical-Intelligence/openpi) | 图像工具取自 `215abfb217dbac7d5f1273282331b9b1866c0479` |
| cuRobo | [NVlabs/cuRobo](https://github.com/NVlabs/curobo) | `78fd485fa82d9b9a063fb4985e371814587e666a` |

原始数据及大模型通过固定版本从 Hugging Face 下载，不重复放进 Git。
模型根目录是 25000 步；`checkpoints/` 中每 2500 步一份。
下载器仅下载根目录 7 个推理文件并校验模型大小和 SHA256，避免下载全部副本。
PaliGemma tokenizer 需要先接受其访问条款；通过 `hf auth login` 在自己的机器登录。

OpenPI 官方仓库提供模型背景及图像工具。本训练入口实际使用未修改的
`lerobot[pi,training]==0.6.1`，不调用 OpenPI 的 JAX 训练器。
为避免引入 openpi-client 的 `numpy<2` 安装约束，仅将其官方图像工具连同许可证
原样收录在 `third_party/openpi/`。该提交是保留的采集环境版本，
尚不能证明与最初训练转换环境的提交完全相同。

## 安装

目标平台为 Linux x86_64，NVIDIA CUDA 12.8、Python 3.12。
原全参数训练设备为 RTX PRO 6000 Blackwell 96GB、24 CPU 核、177GB RAM；
batch32 的最低显存尚未测量。cuRobo 编译需要 Git、C++ 编译器及 CUDA Toolkit。
OrcaStudio 渲染还需要图形显示环境和对应资源包。详细版本见
[版本说明](docs/VERSIONS_AND_HANDOFF.md)。

```bash
git clone --branch feature/pi05-v9-blue-box-200-full-finetune \
  https://github.com/JasonZhw/OrcaPlayground.git
cd OrcaPlayground/examples/euler/12_pi05_blue_box
bash scripts/bootstrap.sh
conda activate orca
python reproduce.py check --profile simulation --strict-versions --cuda
```

已有 `orca` 环境时，不要再次运行创建环境的脚本：

```bash
conda activate orca
python -m pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt -r requirements-simulation.txt
python -m pip install --no-build-isolation \
  -c requirements-build.txt \
  'nvidia-curobo[cu12-torch] @ git+https://github.com/NVlabs/curobo.git@78fd485fa82d9b9a063fb4985e371814587e666a'
python -m pip check
```

严格复现训练依赖时，在另一台训练机器使用
`bash scripts/bootstrap.sh --training-only`。
该模式安装 `requirements-training.txt`，保留原Pillow12.3.0；
仿真模式遵循OrcaLab26.9.1要求使用Pillow12.0.0。
两种Pillow不能在同一环境同时锁定，不要将两份profile一起安装。
训练入口与参数相同，精确环境复现实验应记录所选profile。

## 使用已有权重做 10 次闭环推理

先通过自己的 Hugging Face 账号取得 PaliGemma 访问权限，再登录并下载：

```bash
hf auth login
python reproduce.py download checkpoint --destination models/pi05-025000
python reproduce.py download tokenizer --destination models/tokenizer
python reproduce.py prepare-scene
```

在有显示环境的终端启动空白 Studio：

```bash
python reproduce.py studio
```

首次初始化完成后，按提示重新启动。登录 OrcaLab，按
[场景与采集](docs/SCENE_AND_COLLECTION.md) 订阅并同步四个资源包。
资源准备好后，在第二个同环境终端进入本目录：

```bash
python reproduce.py studio --load
python reproduce.py check-scene --output outputs/scene-check
```

先人工检查 `outputs/scene-check/` 中的头部与右腕画面、物品和木框。
检查器只验证相机帧有效，不识别物品或证明纹理资源一致。然后运行：

```bash
python reproduce.py evaluate \
  --checkpoint models/pi05-025000 --tokenizer models/tokenizer \
  --output outputs/eval-pad-10 --trials 10 --image-mode pad
python reproduce.py audit \
  --evaluation outputs/eval-pad-10 --output outputs/eval-pad-10-audit.json
```

输出为 `summary.json` 和每次试验的 `cameras.mp4`、`trajectory.npz`、
`predicted_chunks.npz`、`result.json`。中止与失败也计入分母。
模型每次预测 50 个绝对动作，执行 50 步后重规划；控制频率 50Hz。
脚本自动处理执行器限幅、相机采样与双相机录像，无需手动操作录屏软件。
成功要求夹起超过 8cm、整个盒子位于框内、接触框底、主动松爪并脱离夹爪，
且速度低于 2cm/s 连续稳定 50 个周期。碰撞或关节跳变只会中止测试。
这是仿真入口，不连接真实机器人。
另有 `provenance.json` 保存实际权重/tokenizer 哈希、代码版本和环境清单，
每次试验保留初始双相机 PNG。审计通过只表示记录一致，
仍须人工审核全部录像；审计报告中的成功率保留失败和中止试验。

统一使用命令中的 `--image-mode pad`：头部与右腕图像保持比例缩放至 224×224，
不足部分补黑边，与训练转换一致。脚本自动完成并记录，无需单独运行预处理命令。
这只影响模型输入，不改变相机视角或电脑录屏设置。

每轮记录实际成功数与总请求数；当前任务的物品和木框位置固定。
结果文件和成功判据见 [评测说明](docs/EVALUATION.md)。

## 复现原始 200 条数据的全参数训练

公开数据来自本项目原始 200 条成功示范，见上方资源表。
在独立的训练环境中完成以下步骤：

```bash
python reproduce.py download raw --destination data/raw_v9
python reproduce.py convert data/raw_v9/raw data/v9
python reproduce.py download base --destination models/pi05-base
python reproduce.py download tokenizer --destination models/tokenizer
python reproduce.py prepare-train \
  --dataset-root data/v9 --dataset-id local/v9 \
  --base-model models/pi05-base --tokenizer models/tokenizer \
  --output-dir outputs/train-v9 --config-out generated_train_config.json
python reproduce.py train \
  --config-path generated_train_config.json --tokenizer models/tokenizer
```

转换将验证全部 200 条成功示范、277084 帧、50Hz 及视频/状态/动作逐帧一致。
每帧左右两路 640×480 RGB 经官方 `resize_with_pad` 转为 224×224，
图像直接嵌入 LeRobot v3.0 Parquet。状态和动作都是 9 维。

训练参数保留原保存配置：全参数、bf16、batch32、AdamW lr2.5e-5、
预热1000步、计划30000步余弦衰减、每2500步保存。
**公开结果使用 25000 步权重，学习率调度仍按 30000 步计算。**
可训练至30000步后选择025000检查点；不要把总步数改为25000并称作同一调度。
便携入口只调整资源位置、tokenizer 来源与日志设置，默认关闭 W&B 和自动上传。
原训练命令、完整参数及续训限制见 [训练配方](docs/TRAINING_RECIPE.md)。

## 自行用 cuRobo 采集

更换场景或物品时，先相应调整布局、采集目标与规划配置，再采集适用于新任务的示范。
下面的命令使用课程自带的 V9 场景。准备并启动场景、完成基础模型与 tokenizer 下载后：

```bash
python reproduce.py collect --output data/new_v9 --count 200
python reproduce.py convert data/new_v9/raw data/new_v9_lerobot --new-collection
python reproduce.py prepare-train \
  --dataset-root data/new_v9_lerobot --dataset-id local/new-v9 --new-collection \
  --base-model models/pi05-base --tokenizer models/tokenizer \
  --output-dir outputs/train-new-v9 --config-out outputs/new_train_config.json
python reproduce.py train \
  --config-path outputs/new_train_config.json --tokenizer models/tokenizer
```

采集脚本包含预抓取、对准、闭爪、双侧接触、抬起、搬运、放置、松爪、撤离和稳定阶段；
录像与动作一一对应。已有成功轨迹可继续采集，连续两次失败会停止。
V9 物品与框位置固定，没有随机摆放泛化数据。
新采集的轨迹长度可能不同，所以转换与训练预检均需加 `--new-collection`。
该新数据不能冒充原始277084帧数据。

## 目录与验证

```bash
python -m unittest discover -s tests -v
python scripts/verify_public.py
python scripts/smoke_physics.py
python -m ruff check --select SLF001 scripts simulation tests
```

`smoke_physics.py` 需要先执行 `prepare-scene`；离线物理检查不提供训练相机画面。
文件职责见 [文件清单](docs/FILE_INVENTORY.md)；第三方资源见 [来源说明](THIRD_PARTY.md)。
运行生成的模型、数据、权重与测试录像均已加入忽略规则。
