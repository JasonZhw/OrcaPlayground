# 原训练入口与完整配方

依据原训练者提供的命令、脚本、配置及环境清单整理。
本文件示例只用通用本地路径；不包含原操作服务器或个人电脑的地址。

## 保存配置入口

`configs/train_config.saved.json` 是原保存配置，包含轻微仿射增强。
将其中 `dataset.root` 与 `output_dir` 改为本机新目录，可使用原训练者确认的入口：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false \
lerobot-train --config_path=train_config.json
```

原配置 `pretrained_revision=null`，原训练者补充确认当时 main 对应
`22c30e5b34f9e7ab738b1b34fc2fc5ecfe2d28f0`。
原 tokenizer 为 `google/paligemma-3b-pt-224`，
对应 `35e4f46485b4d07967e7e9935bc3786aad50687c`。
直接使用旧配置会重新解析当前远程资源；便携入口固定下载资源并显式指定本地 tokenizer。

## 实际自定义入口

原 `train_stage.py` 的环境变量如下，仅把原 Colab 路径换成通用本地路径：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false \
PRETRAINED=lerobot/pi05_base DATA_ROOT=./data/v9 OUT_DIR=./outputs/run \
STEPS=30000 BATCH=32 LR=2.5e-5 WARMUP=1000 DECAY_STEPS=30000 WEIGHT_DECAY=0.01 \
GRAD_CKPT=true SAVE_FREQ=2500 WANDB_RUN=pi05-v9-single-target-200 \
NORM='{"VISUAL": "IDENTITY", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"}' \
python reference/training/train_stage.py
```

该入口启用 W&B，原运行采用单机单卡，无多卡启动器，无梯度累积。
不设置实际环境变量而直接运行脚本默认值，可能得到不同配方：
`GRAD_CKPT` 默认 false，`NORM` 默认 QUANTILES。原运行明确覆盖为 true / MEAN_STD。
`run_train.py` 负责 Colab 云盘同步和清理，不是必要训练入口；
本复现包不收录该同步程序，避免引入远程存储与清理行为。

## 调度与模型选择

计划 30000 步，1000 步预热，之后余弦衰减，25000 步人工停止。
25000 步时仍按 30000 步调度，并未衰减到末端学习率。
仅将 `STEPS=25000`，即使另设 `DECAY_STEPS=30000`，
也不能未经核对当作同一训练调度；便携配置固定原 30000 步计划。
完成至 30000 后选取 025000 权重也是可审查的复现方式。

每 2500 步保存。公开模型根目录与 `checkpoints/025000` 的 7 个文件一致，
但没有完整优化器/调度器/RNG 状态。`train_stage.py` 中的 `RESUME_CONFIG`
需要另有完整本地 checkpoint，不能只凭公开模型文件承诺无缝续训。

## 与便携入口的区别

| 项目 | 原运行 | 便携复现 |
| --- | --- | --- |
| 模型来源 | 当时 main | 固定 revision 的本地快照，或固定 Hub revision |
| tokenizer | 处理器使用仓库名称 | 固定 revision 快照，处理器指向本地目录 |
| W&B | 开启 | 默认关闭 |
| 路径与运行 ID | Colab 本地路径及原 ID | 用户指定路径与新运行标识 |
| 全参数开关、优化器、调度、增强 | 原保存配置 | 保留原值 |
| 图像转换 | 官方 resize_with_pad | 同一接口；原客户端提交仍待确认 |

便携 `train.py` 对当前进程的处理器工厂做单一适配，合并原 overrides 后设置 tokenizer 路径。
不修改已安装的 LeRobot 源文件。该适配尚未在真实 GPU 训练中验证，不能声称数值逐步一致。

## 原始数据转换

原训练者给出的命令（按当前 Hub main 下载，适合阅读原流程）：

```bash
hf download xiaojia666/g1-blue-intel-single-target-200 --repo-type dataset --local-dir raw_v9
python reference/training/convert_raw.py raw_v9/raw data/v9
```

实际复现建议用 README 的固定 revision 下载入口。
原脚本和便携脚本都从 raw 直接写 LeRobot v3.0：
成功过滤、50 Hz、按录像左右切分、RGB、保持比例补黑边到 224×224、
state/action float32[9]、图像写 Parquet、全部 200 条参与训练。
便携版本补充显式错误、输出覆盖保护、FFmpeg 失败清理和总数检查；
数值与图像转换逻辑保留，尚未全量运行并对比最终数据统计。

严格匹配训练包版本时使用 `requirements-training.txt`；
OrcaLab仿真profile因GUI依赖需要Pillow12.0.0，与原训练Pillow12.3.0不同，
不能混用后声称环境完全一致。
