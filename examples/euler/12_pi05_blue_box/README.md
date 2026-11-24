# V9 蓝色物品夹取：200 条示范与 pi0.5 全参数微调

任务：G1 固定底座，右臂夹起蓝色 Intel 盒子，搬运到木框内并主动松爪。
本示例提供原始数据下载、cuRobo 自动采集、Euler 场景、数据转换、完整训练配置、
权重下载与闭环录像评测。训练使用 LeRobot 的 pi0.5 实现，全参数更新。

建议首先下载已有权重进行推理，然后复现原始 200 条数据的训练。
重新采集属于另一条可选流程，不需要为下载权重测试先采集 200 条。
验证范围与尚未执行的步骤见 [复现状态](docs/REPRODUCTION_STATUS.md)。

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
python reproduce.py check --simulation --cuda
```

已有 `orca` 环境时，不要再次运行创建环境的脚本：

```bash
conda activate orca
python -m pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt -r requirements-simulation.txt
python -m pip install --no-build-isolation \
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

```bash
python reproduce.py download checkpoint --destination models/pi05-025000
python reproduce.py download tokenizer --destination models/tokenizer
python reproduce.py prepare-scene
```

先阅读 [场景与采集](docs/SCENE_AND_COLLECTION.md)，登录 OrcaLab 并同步布局列出的四个资源包。
在有显示环境的终端启动：

```bash
python reproduce.py studio
```

等待 OrcaStudio 加载完成，在第二个同环境终端进入本目录：

```bash
python reproduce.py studio --load
python reproduce.py evaluate \
  --checkpoint models/pi05-025000 --tokenizer models/tokenizer \
  --output outputs/eval-pad-10 --trials 10 --image-mode pad
```

输出为 `summary.json` 和每次试验的 `cameras.mp4`、`trajectory.npz`、
`predicted_chunks.npz`、`result.json`。中止与失败也计入分母。
模型每次预测 50 个绝对动作，执行 50 步后重规划；控制频率 50Hz。
成功要求夹起超过 8cm、整个盒子位于框内、接触框底、主动松爪并脱离夹爪，
且速度低于 2cm/s 连续稳定 50 个周期。碰撞或关节跳变只会中止测试。
这是仿真入口，不连接真实机器人。

历史记录采用直接拉伸图像；对照该预处理时使用新的输出目录：

```bash
python reproduce.py evaluate \
  --checkpoint models/pi05-025000 --tokenizer models/tokenizer \
  --output outputs/eval-legacy-10 --trials 10 --image-mode legacy-stretch
```

新脚本的保护器与历史保护器有区别，不能视为原始分数的逐项复现。
默认 `pad` 与训练转换的图像几何一致，尚无此新入口的成功率。
历史固定场景记录及限制见 [评测说明](docs/EVALUATION.md)。

## 复现原始 200 条数据的全参数训练

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

准备并启动同一场景后：

```bash
python reproduce.py collect --output data/new_v9 --count 200
python reproduce.py convert data/new_v9/raw data/new_v9_lerobot --new-collection
python reproduce.py prepare-train \
  --dataset-root data/new_v9_lerobot --dataset-id local/new-v9 --new-collection \
  --base-model models/pi05-base --tokenizer models/tokenizer \
  --output-dir outputs/train-new-v9 --config-out outputs/new_train_config.json
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
