```text
π0.5 依赖层与运行入口
├── 保留现有环境
│   ├── 唯一 .venv/bin/python
│   ├── Torch 2.14.1+cu130、NumPy 2.2.6、OrcaLab 不变
│   └── 训练进程单独使用 .venv/pi05-deps
├── 安装准备
│   ├── 离线构造命令
│   ├── 只读核对发行版元数据
│   └── 显式安装；中断后核对快照再重试
├── 必需接口
│   ├── JAX 0.5.3、CUDA 12.9、cuDNN 9.10.2
│   ├── Transformers 4.53.2
│   └── 固定 LeRobot revision 的数据集 API
└── 验证与训练
    ├── CPU 数据写读与 G1 变换恢复
    ├── 协调 GPU 后验证小规模运算
    └── 真实成功回合、独立统计量、JAX LoRA 训练与推理
```

# 唯一 .venv 的 OpenPI JAX LoRA 入口

工作目录为 `/home/pc/ZXJ/SY`。此入口不改 `software/openpi`，不创建新的 venv，不使用 `pi05-venv`。OpenPI 固定 commit 为 `215abfb217dbac7d5f1273282331b9b1866c0479`。
2026-10-13 已有安装任务完成，标准 wrapper 的 CPU verify 与 GPU 小运算 verify
通过，官方基座全文件校验、CPU 参数恢复、形状检查和 tokenizer 检查通过。
当前主线已完成 1500 步真实训练、完整 1499 检查点保存和全部五回合真实推理，成功率 1/5；正在继续低学习率微调。下文早期验证段落保留各自当时的范围，实时结果见 `runtime/reports/pi05-task-progress.json` 和 `tools/medication_sorting/README.md`。

## 显式运行安装入口

以下三条命令依次用于查看计划、获取元数据、执行安装。默认入口和 `install` 不带 `--execute` 均只打印命令，不运行 pip，也不创建 target。

```bash
cd /home/pc/ZXJ/SY
.venv/bin/python tools/medication_sorting/setup_pi05.py --dry-run
.venv/bin/python tools/medication_sorting/setup_pi05.py metadata
.venv/bin/python tools/medication_sorting/setup_pi05.py install --execute
```

安装分两步，均使用 `.venv/bin/python -m pip install --no-deps --target /home/pc/ZXJ/SY/.venv/pi05-deps`：先安装从 pinned `uv.lock` 展开的精确 registry 版本，再安装 pinned LeRobot Git 源码。第一步限定 `--only-binary=:all:`；第二步只构建 LeRobot wheel，其 Poetry 构建依赖使用 pip 的临时构建目录。完整 argv、每个精确版本、lock SHA256 和引用来源均在 dry-run JSON 中。OpenPI 和 client 直接从只读源码导入，无须安装 distribution。

入口丢弃继承的 pip 配置和 `PYTHONPATH`，避免把隐含的 `--user`、索引或安装路径带进命令。受保护快照记录基环境 Torch、NumPy、OrcaGym、OrcaLab、OrcaLab PySide 和 OpenCV 的版本、METADATA 与 RECORD 哈希；安装前后核对。快照验证的是包元数据，实际功能仍需后续验证。

安装中断或网络失败后，保留 target，直接重试：

```bash
.venv/bin/python tools/medication_sorting/setup_pi05.py install --execute --retry
```

重试只接受本入口留下的 `installed:false` manifest，要求固定 commit、lock、精确 pins、路径及受保护快照都一致。它为同一安装命令补上 `--upgrade --no-deps`，修复 pip `--target` 默认会跳过的已有 partial 包目录，并复用 pip 下载缓存。不需要清理依赖层。未记录来源的非空 target、完整安装、快照变化或不同版本计划均拒绝覆盖；partial 层不能通过 wrapper 启动。

## 精确版本与下载量

| 组件 | 训练层版本 | 依据或差异 |
|---|---|---|
| jax / jaxlib / jax-cuda12-plugin / jax-cuda12-pjrt | 全部 `0.5.3` | 官方 OpenPI 固定版本 |
| flax / orbax-checkpoint | `0.10.2` / `0.11.13` | 官方固定版本 |
| ml-dtypes / tensorstore | `0.4.1` / `0.1.74` | 官方 override |
| NumPy | `1.26.4` | 仅 target；基环境仍为 `2.2.6` |
| Transformers | `4.53.2` | 官方固定版本，JAX 公共模块也必须导入 |
| Torch | 复用 `2.14.1+cu130` | 官方为 `2.7.1`，明确偏离 |
| torchvision | `0.29.1` | PyPI 声明 `torch>=2.14.0`；替代官方锁 `0.22.1` |
| wrapt | `1.17.3` | 官方锁 `1.14.1` 没有 Python 3.12 wheel；满足 dm-tree/deprecated 约束 |
| LeRobot | `0.1.0`，Git rev `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5` | 必须核对 direct_url revision，不能用最新版替代 |

| CUDA12 wheel | 精确版本 |
|---|---|
| nvidia-cublas-cu12 | `12.9.1.4` |
| nvidia-cuda-cupti-cu12 | `12.9.79` |
| nvidia-cuda-nvcc-cu12 | `12.9.86` |
| nvidia-cuda-nvrtc-cu12 | `12.9.86` |
| nvidia-cuda-runtime-cu12 | `12.9.79` |
| nvidia-cudnn-cu12 | `9.10.2.21` |
| nvidia-cufft-cu12 | `11.4.1.4` |
| nvidia-cusolver-cu12 | `11.7.5.82` |
| nvidia-cusparse-cu12 | `12.5.10.65` |
| nvidia-nccl-cu12 | `2.26.5` |
| nvidia-nvjitlink-cu12 | `12.9.86` |

只读 PyPI 核查的 137 个 registry 包压缩 wheel 合计约 **3.02 GiB（3.24 GB）**，不重复下载 Torch。估算不含 LeRobot Git、Poetry 临时构建依赖、模型权重、数据集和训练 checkpoint。建议预留 **10–12 GB** 给解压后的依赖层、pip 缓存和临时文件；模型与数据另外预留。`metadata` 输出逐包 wheel 名称、大小、SHA256、Requires-Dist，并核对基础约束及 JAX CUDA extras；不是全量 LeRobot 硬件依赖的 `pip check`。

## 必须安装的 API 与可以省去的依赖

Transformers `4.53.2` 提供 `AutoProcessor`、`GemmaForCausalLM` 和 `PaliGemmaForConditionalGeneration`。OpenPI 的 JAX 公共模型模块仍导入这些符号，不能因为选择 JAX 就删除 Transformers。JAX 路线不复制 `transformers_replace`，也不覆盖基环境源码。

固定 LeRobot 的接口为：

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
# LeRobotDataset.create(...), dataset.add_frame(...), dataset.save_episode()
```

数据路径需要 datasets、pyarrow、Pillow、av、torchvision.transforms、draccus、jsonlines、NumPy、Torch、huggingface-hub 及其传递依赖。本任务采用 image 数据；`save_episode` 即落盘，没有使用新版 `consolidate/finalize`。av 保留给官方模块的导入和 pyav 路径，不需要 torchcodec。

不安装 gym-aloha、gymnasium、diffusers、cmake、h5py、numba、pynput、pymunk、rerun-sdk、zarr、各硬件驱动 extras 或 RLDS/TensorFlow/dlimp。它们属于 LeRobot 的其他功能，不属于此任务的数据路径；通过 `--no-deps` 有意省去。因而全环境 `pip check` 会看到 LeRobot 未满足的硬件依赖，以及 OpenPI/Torch、Orca/NumPy 的混合约束，不能把它当成完整官方环境。

## wrapper 与安装后的验证

```bash
tools/medication_sorting/pi05_python.sh --dry-run --cpu \
  -m tools.medication_sorting.setup_pi05 verify
tools/medication_sorting/pi05_python.sh --cpu \
  -m tools.medication_sorting.setup_pi05 verify
```

wrapper 固定同一解释器，仅在子进程中把 target、项目和 OpenPI/client 源码加入 `PYTHONPATH`，禁用 user site，去除 `LD_LIBRARY_PATH`、`LD_PRELOAD`、`PYTHONHOME`。`--cpu` 隐藏 GPU 并固定 JAX CPU backend。它保留 argv，支持其他工作目录，禁止会绕过依赖层的 `-I/-E/-S`，以及误用 `-m pip/venv/ensurepip`。只用它运行适配、转换和训练进程，日常仿真继续使用原入口。

本机 Torch 2.14 在默认 `_preload_cuda_deps` 路径按 `sys.path` 先加载训练层
NCCL 2.26.5，其缺少 Torch 所需的 `ncclCommResume`，因此首次 CPU verify
在导入 Torch 时失败。基础 Torch 单独导入正常，`LD_DEBUG=libs` 显示错误进程
实际加载的是 target 的 `libnccl.so.2`。wrapper 现在只为当前子进程固定
`TORCH_USE_RTLD_GLOBAL=1`，让 Torch 采用其原生 RPATH 路径，避开这一预加载
冲突。基础环境、Torch 包和 OpenPI 源码未修改。此设置使 Torch 符号在本进程
全局可见，不能解释成 CUDA12/13 完全隔离；本机小 GPU 检查已通过，真实训练
峰值和模型推理仍需独立验收。两项回归测试先复现失败，再在修复后通过。

CPU `verify` 实际检查 NumPy/Torch bridge、Transformers API、官方 LeRobot 四帧图像/Parquet 写读、动作 chunk、G1 七关节 delta/32 维 padding/恢复，以及外置 LoRA 配置导入。它不下载 tokenizer/模型，也不执行训练或推理，报告会保留 `training_step_tested:false`、`inference_tested:false`。

GPU 小运算检查在与主线程协调 GPU 时段后执行：

```bash
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.05 \
tools/medication_sorting/pi05_python.sh \
  -m tools.medication_sorting.setup_pi05 verify --gpu
```

此入口只检查 GPU 可见性、32×32 bf16 GEMM 和微型卷积，不是训练或 benchmark。默认设置 `XLA_PYTHON_CLIENT_PREALLOCATE=false` 和内存比例 `0.75`，可由当前子进程环境覆盖；这不保证训练峰值显存。

## 真实成功数据、统计量、训练和推理

以下路径仅为命令模板，将 `/absolute/raw_success_episodes` 替换为主线程完成释放后的成功完整回合目录。主线程确认真实采集频率为 **5 Hz**，转换使用 `--fps 5`，动作 chunk 的时间间隔为 0.2 秒；horizon 10 覆盖约 2 秒。数据语义保持七个右臂关节绝对位置 rad 加 closure，G1 不复用 DROID 统计量。只有 RGB 质量、时间同步和完整成功条件均合格的回合才进入转换；主线程安排至少 20 个合格回合后训练，一步验收也由主线程在首合格回合和释放 GPU 后安排。

```bash
export HF_LEROBOT_HOME=/home/pc/ZXJ/SY/data/medication_sorting/lerobot
export OPENPI_DATA_HOME=/home/pc/ZXJ/SY/runtime/cache/openpi
tools/medication_sorting/pi05_python.sh --cpu \
  -m tools.medication_sorting.convert_episodes \
  --input /absolute/raw_success_episodes \
  --output "$HF_LEROBOT_HOME/local/g1_medication" \
  --repo-id local/g1_medication --fps 5
tools/medication_sorting/pi05_python.sh --cpu \
  -m tools.medication_sorting.pi05_adapter --repo-id local/g1_medication norm-stats
tools/medication_sorting/pi05_python.sh \
  -m tools.medication_sorting.pi05_adapter --repo-id local/g1_medication \
  --batch-size 1 --action-horizon 10 train -- --exp-name=g1_trial \
  --weight-loader.params-path=/home/pc/ZXJ/SY/data/medication_sorting/pretrained/pi05_base/params
tools/medication_sorting/pi05_python.sh \
  -m tools.medication_sorting.pi05_adapter --repo-id local/g1_medication \
  --action-horizon 10 serve -- --policy.dir=/absolute/g1_checkpoint --port=8000
```

`HF_LEROBOT_HOME/repo_id` 必须与转换输出对应，官方 loader 不接受任意输出 root。
本地基座经完整校验后，训练通过 `--weight-loader.params-path` 指向它的 `params`
目录，避免再次获取同一基座；不提供这个参数时，配置默认使用官方 `gs://` 路径。
PaliGemma tokenizer 是单独的 `gs://big_vision/paligemma_tokenizer.model`，基座
manifest 不包含它；已提前缓存，沿用上述 `OPENPI_DATA_HOME` 即可直接使用。
CPU 离线检查已拦截下载函数，确认真实 model transforms 从缓存编码成功。
真实端到端验收应包括：成功
回合转换、该数据的 norm stats、真实权重的一步 LoRA 训练与 checkpoint 保存、
加载该 checkpoint 推理并恢复 8 维输出。可以在训练参数中加入
`--num-train-steps=1 --save-interval=1 --log-interval=1` 做首次验收；小
GEMM/卷积通过不能替代这条验收链。本次独立任务不执行这条训练命令。

### Tyro 参数与 checkpoint 目录

适配器参数放在模式前，例如 `--repo-id`、`--batch-size`、`--action-horizon`。
官方脚本参数放在模式后面的 `--` 之后。适配器会自动注册
`pi05_g1_medication_sorting_lora`，并为 serve 自动补上 `policy:checkpoint` 和
`--policy.config`，不需要用户重复传入。
实际 Tyro 解析确认 `--port`、`--default-prompt`、`--env`、`--record` 等根参数
必须位于 `policy:checkpoint` 之前；适配器会自动调整这些参数的位置，用户
仍按以上模板传参。修复前 `serve -- --policy.dir=... --port=8000` 会报
`Unrecognized options: --port=8000`，三项先失败后通过的回归测试覆盖此问题。

| 入口 | 官方参数 | 含义 |
|---|---|---|
| train | `--exp-name=g1_trial` | 必须提供，决定实验目录 |
| train | `--weight-loader.params-path=/absolute/pi05_base/params` | 本地官方基座参数目录 |
| train | `--num-train-steps=1` | 一步验收模板，尚未实际执行 |
| train | `--save-interval=1`、`--log-interval=1` | 保存和日志间隔 |
| train | `--resume` | 恢复实验；不得同时使用 `--overwrite` |
| serve | `--policy.dir=/absolute/experiment/STEP` | 某个编号 checkpoint 的根目录 |
| serve | `--port=8000`、`--default-prompt=...` | WebSocket 端口和默认任务文本 |

以下格式来自 pinned `train.py`、`training/checkpoints.py` 和 `policy_config.py`
源码；是未来输出的目录契约，不代表已有训练产物。

```text
data/medication_sorting/checkpoints/
└── pi05_g1_medication_sorting_lora/
    └── g1_trial/
        └── STEP/
            ├── params/          # Orbax 参数，可供 JAX 推理加载
            ├── train_state/     # 优化器、步数等训练状态
            └── assets/
                └── local/g1_medication/norm_stats.json
```

训练保存循环以从 0 开始的循环编号命名目录；`--num-train-steps=1` 的最终保存
目录是 `g1_trial/0`。serve 的 `--policy.dir` 应指向这一层，同时存在 `params`
和 G1 自有 `assets`；不要指向 `params` 子目录、实验父目录或官方基座根目录。
训练和 serve 的 `--action-horizon` 必须一致。当前官方 `initialize_checkpoint_dir`
把只有编号 `0` 的实验视为不能恢复的首轮状态，因此一步验收后再次用 `--resume`
会重新初始化；不要把它视为已验证的断点续训。

依赖安装完成后，用 CPU wrapper 查看实际 Tyro 帮助，不初始化模型：

```bash
tools/medication_sorting/pi05_python.sh --cpu \
  -m tools.medication_sorting.pi05_adapter train -- --help
tools/medication_sorting/pi05_python.sh --cpu \
  -m tools.medication_sorting.pi05_adapter serve -- --help
```

### 官方基座全文件复验

`data/medication_sorting/reports/pi05-download-manifest.json` 包含 29 个文件，
总长 12,441,749,581 bytes。全部有 CRC32C，其中 23 个有 MD5；其余 6 个是
GCS 组合对象。旧下载器对这些组合对象只核对长度，完成标记中的
`md5_verified:true` 不能证明全文件校验成功。

安装层与已有下载均结束后，执行不访问网络的完整复验：

```bash
tools/medication_sorting/pi05_python.sh --cpu \
  -m tools.medication_sorting.download_pi05 --verify-only
```

修复后的 `download-verification.json` 逐文件保存实际长度、MD5/CRC32C 与
manifest 的对照。只有 `verified_files == total_files == 29` 且
`all_files_verified:true` 才表示基座完整。由于官方缺少 6 个 MD5，正确结果是
`md5_verified:false`、`md5_verified_files:23`，以及
`crc32c_verified:true`、`crc32c_verified_files:29`。复验不读取模型，不占用 GPU。
已有进程加载的是旧版本代码，文件完成后必须运行以上离线复验，不能沿用旧标记。

2026-10-13 11:49 实际基座验收已通过：29/29 个文件长度与 CRC32C 匹配，
23/23 个可用 MD5 匹配，无 `.partial` 残留。由于依赖安装仍在继续，先用唯一
`.venv/bin/python` 将已有 pip 下载缓存中的 `google-crc32c==1.7.1` wheel
解压到临时目录，仅为本次检查加入 `sys.path`；检查后自动移除临时目录。
wheel SHA256 已与 PyPI 官方 metadata 对照，没有另起安装或修改基础/target。
安装就绪后仍会经 CPU wrapper 复验入口。

证据：`runtime/reports/pi05-crc32c-cache-check.json`、
`runtime/reports/pi05-download-full-validation.json`、
`runtime/logs/pi05-download-verify.log` 和基座目录的 `download-verification.json`。
manifest SHA256 为
`96154ede7a573c72363c652ecfb7724289c3bfd6ab36c87d697c155449c693f0`。
这里只证明本地官方基座文件完整；没有加载模型、执行训练或推理。

## 5090 32GB 与依赖冲突结论

建议优先 JAX π0.5 LoRA，配置为 `gemma_2b_lora` + `gemma_300m_lora`、batch 1、horizon 10、EMA 关闭。官方仓库估算 LoRA 需要 >22.5 GB，全参需要 >70 GB。当前仿真约 5 GB，32 GB 卡可能容纳小 batch LoRA，但实际峰值必须测量，训练时段应由主线程统一安排。

明确的冲突包括：基环境 NumPy `2.2.6` 与 OpenPI `<2`，OrcaGym `numpy==2.2.6` 与训练层 `1.26.4`，OpenCV `5.0.0.93` 的 NumPy>=2 与训练层 NumPy1，现有 Torch `2.14.1+cu130` 与官方 `2.7.1`，以及官方 torchvision `0.22.1` 与现有 Torch。Python 依赖层处理进程内的版本选择，不能隔离原生 CUDA 动态库。

Torch 的 RPATH 指向基环境 CUDA13/cuDNN/NCCL；训练层另有 CUDA12，而 cuDNN、NCCL 存在相同 SONAME。这是实际混合环境的风险，不能用 PYTHONPATH 或元数据一致性证明完全隔离。必须以本机实际 JAX GPU、LeRobot API、真实训练和推理结果验收，不能宣称“完全官方环境”或“5090 已通过”。

版本依据：[JAX CUDA12 plugin metadata](https://pypi.org/pypi/jax-cuda12-plugin/0.5.3/json) 允许 cuDNN>=9.1,<10、nvcc>=12.6.85；[cuDNN 9.10.2 支持矩阵](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.10.2/reference/support-matrix.html) 列出 CUDA12.8/12.9、Blackwell、SM12.0 和 Linux driver>=570.26；[CUDA12.9 Update1 发布说明](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-toolkit-release-notes/index.html) 的配套驱动下界为575.57.08；[torchvision0.29.1 metadata](https://pypi.org/pypi/torchvision/0.29.1/json) 声明 Torch>=2.14.0。以上是选型依据；本机已实际验证 GPU 小运算，尚未验证真实训练和推理。

## 2026-10-13 实际运行验收

已有安装 PID 1774252 完成，两步安装成功，manifest 为 `installed:true`；
LeRobot `direct_url.json` 的 commit 为指定 revision。受保护包快照与安装前
一致，基础 NumPy 仍为 2.2.6、Torch 发行版仍为 2.14.1，运行版本为
2.14.1+cu130；训练层 NumPy 为 1.26.4、PyAV 为 17.0.0。主线程另行安装的
基础 PyAV 14.2.0 保留，wrapper 实际从 target 导入 PyAV。

| 验收 | 真实结果 | 证据 |
|---|---|---|
| CPU verify | bridge、四帧 LeRobot 写读、G1 变换、配置导入通过 | `runtime/logs/pi05-cpu-verify.log` |
| GPU verify | cuda:0、32×32 bf16 GEMM、微型 bf16 conv 通过 | `runtime/logs/pi05-gpu-verify.log` |
| 基座文件 | 29 CRC32C、23 可用 MD5 全通过，12,441,749,581 bytes | `runtime/reports/pi05-download-full-validation.json` |
| wrapper 全文件复验 | 返回码 0、29/29 校验通过 | `runtime/logs/pi05-download-wrapper-verify.log` |
| Orbax 真实恢复 | 51/51 数组恢复到 NumPy，shape/dtype/有限值通过 | `runtime/reports/pi05-checkpoint-restore.json` |
| 参数兼容 | 基座 51 项匹配 LoRA 模型 71 项，其余 20 项均为 LoRA；horizon 10 与 50 参数形状一致 | `runtime/reports/pi05-checkpoint-shapes.json` |
| tokenizer | 4,264,023 bytes，SentencePiece 加载和 π0.5 状态提示编码通过 | `runtime/reports/pi05-tokenizer-validation.json` |
| Tyro 解析 | 本地权重、一步参数、serve checkpoint/port 解析通过 | `runtime/reports/pi05-cli-parameters.json` |
| 离线训练初始化 | 官方 train 导入、优化器、抽象训练状态和缓存 model transforms 通过，未分配模型数组 | `runtime/reports/pi05-train-init-audit.json` |
| 相关回归测试 | 120 passed in 2.39s | `runtime/logs/pi05-layer-tests.log` |

权重展开后为 13,413,735,488 bytes，最大单数组 4,831,838,208 bytes。
此次按数组依次恢复并释放，避免同时驻留整套权重；形状比较使用 CPU
`jax.eval_shape`，只建立抽象参数形状，没有分配模型权重或 GPU 数组。
恢复采用同一 Orbax `PyTreeCheckpointer` 和 NumPy restore 路径，未运行整套
`CheckpointWeightLoader.load`、优化器初始化或一步训练。最终训练入口仍需
实际加载、设备放置和 checkpoint 写入验收。

tokenizer 使用 pinned `PaligemmaTokenizer` 同一官方 URL 和 `maybe_download`
缓存函数，cache root 为 `runtime/cache/openpi`。SHA256 为
`8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6`。
这是本地文件摘要，不是官方公布的预期哈希；官方下载、加载和编码检查成功。
后续入口务必沿用上面的 `OPENPI_DATA_HOME`，否则会查询另一个缓存目录。

可复跑 CPU 离线初始化诊断：

```bash
OPENPI_DATA_HOME=/home/pc/ZXJ/SY/runtime/cache/openpi \
tools/medication_sorting/pi05_python.sh --cpu runtime/reports/pi05-train-init-audit.py
```

该诊断导入官方 `train.py`，实际解析训练参数，然后调用
`init_train_state(..., resume=True)` 的抽象分支，只检查初始化图、优化器和
sharding；未运行 `main`、数据 loader、权重 loader、训练或 checkpoint 恢复。
所有训练状态数组均为 `jax.ShapeDtypeStruct`。配置含 71 个参数数组，
其中 51 个可训练数组共 466,957,072 个元素：20 项为 LoRA，其余为视觉编码器
和动作/时间投影头。这是上游 `get_freeze_filter()` 的实际行为，真实显存应按
此配置验收。G1 自有 norm stats 当前尚未生成，预期路径为
`data/medication_sorting/openpi_assets/pi05_g1_medication_sorting_lora/local/g1_medication/norm_stats.json`。

源码复查确认训练和推理共用 G1 数据变换及 model transforms，π0.5 启用
`discrete_state_input=True`。默认基座 horizon 为 50；horizon 10 改变动作序列长度，
flow timestep 使用正弦编码和固定宽度 MLP，动作位置由有效 token 的累计位置产生，
没有依赖 horizon 50 的可学习动作位置参数，和实际抽象形状比较结论一致。
实际 5 Hz 的 horizon 10 取 `t/fps`，即 0 到 1.8 秒的 10 个采样点。

双 RGB 对应 `base_0_rgb=True`、`right_wrist_0_rgb=True`，补零的
`left_wrist_0_rgb=False`。官方预处理保留这些 mask，prefix 将每个相机的
mask 扩展到图像 tokens，attention 同时屏蔽无效 query/key；训练和推理一致。
上游仍会计算补零左腕图像的视觉编码，因此 mask 不能视为跳过第三路计算。
这些是源码契约检查，未执行模型 forward。

验收只说明依赖、基座和参数准备情况；所有报告保留
`training_step_tested:false`，未创建训练 checkpoint，未启动策略服务或推理。

## 早期离线实现验证证据

新增测试先观察缺失入口、partial 快照/重试和依赖元数据检查失败，再完成实现。以下命令实际返回 **156 passed in 2.31s**，包含安装入口 48 项、G1 adapter/转换 60 项以及原药品分类工具 48 项。

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu \
.venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/test_setup_pi05.py tests/test_pi05_adapter.py tools/medication_sorting/tests
bash -n tools/medication_sorting/pi05_python.sh
```

Shell 语法和 Python AST 检查通过；`git -C software/openpi status --porcelain` 为空，HEAD 与 pinned commit 一致。只读 PyPI 核查返回 137 个兼容 wheel、3,237,314,239 bytes（3.015 GiB），候选 registry/JAX CUDA extras 约束冲突为空。测试中的安装调用被替换为中断/完成模拟，只向 pytest 临时目录写 manifest，不运行真实 pip。基环境只读核查仍为 Torch `2.14.1+cu130`、NumPy `2.2.6`、OrcaGym/OrcaLab `26.9.1`。
