# 当前示范转为 LeRobot v2.1 / π0.5 训练数据

本次在 `feature/Synthetic_Data` 上新增转换工具和训练适配，未修改录制或运控行为。

## 本地交付

- 原始数据：`/home/jason77/SY/datasets/tray_transfer/`，保留不变。
- LeRobot 数据：`/home/jason77/SY/datasets/lerobot/local/g1_lightblue_box_tray_v1/`。
- 云端交付目录：`/home/jason77/SY/exports/pi05_tray_pilot_v1/`。
- 云端压缩包：`/home/jason77/SY/exports/pi05_tray_pilot_v1.tar.gz`。
- 云端步骤：交付目录的 `README_云端开始.md`。

包含 8 条用户认可的示范、26,617 帧，保持 50 Hz、头部/右腕两路相机、15 维实测状态和 9 维绝对动作。
三种英文任务表述通过 `task_index` 对应每条示范。没有加入离线增强，没有随机划分验证集。
7 条包含右臂复位；另一条保留其用户认可的任务完成状态、滑移诊断和未复位标记。

原录制 `training_ready=false` 表示原格式尚未转换，不应据此排除成功示范。
新数据的 `validation.json` 说明格式和配对检查结果；它不代表已训练或已验证闭环成功率。

## 转换与回读

转换环境位于 `/home/jason77/SY/.venv-tray-lerobot`，基于 `orcalab_9_1` 建立，新增依赖隔离安装。
原仿真环境未升级。固定 LeRobot `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`，与固定 OpenPI 版本一致。
本地实际测试的主要读取依赖是 torch 2.7.1+cpu、torchvision 0.22.1+cpu、datasets 3.6.0、PyArrow 20.0.0、PyAV 15.1.0。

在仓库根目录执行；输出目录已经存在时，转换器会拒绝覆盖。新增数据应维护新的显式清单并选择新的输出目录：

```bash
cd /home/jason77/SY/OrcaPlayground
/home/jason77/SY/.venv-tray-lerobot/bin/python -m examples.euler.tray_transfer.lerobot_export \
  --manifest examples/euler/tray_transfer/lerobot_manifest_v1.json \
  --source-root /home/jason77/SY/datasets/tray_transfer \
  --output /home/jason77/SY/datasets/lerobot/local/g1_lightblue_box_tray_v1
```

```bash
HF_HUB_OFFLINE=1 /home/jason77/SY/.venv-tray-lerobot/bin/python \
  -m examples.euler.tray_transfer.validate_lerobot \
  --root /home/jason77/SY/datasets/lerobot/local/g1_lightblue_box_tray_v1 \
  --source-root /home/jason77/SY/datasets/tray_transfer
```

写入使用官方 LeRobot API；两路 PNG 压缩为 H.264 CRF18 视频。回读逐帧检查数值、视频数量与时间戳，
比较关键画面，验证 50 步动作块不会跨 episode 取值。转换时间戳来自仿真帧号，不使用慢速渲染的墙钟时间。

## 模型契约

`observation.state` 为真实的右臂 7 维＋夹爪 8 维，不能从动作重建。
`action` 为右臂 7 个绝对目标＋连续夹爪参数 a、d，展开为 `[0,0,a,d,0,0,-a,d]`。
使用录下来的有效 nominal action，不使用锁存前的 requested_action，也不把夹爪改成二值开关。

[训练适配](pi05_training/README.md) 使用 JAX π0.5 LoRA，保留绝对动作，分别归一化 state15/action9，再补到 32 维。
头部对应 `base_0_rgb`，右腕对应 `right_wrist_0_rgb`，左腕占位图的 mask 为 False。
输出去掉补零部分，保留 9 维；云端 `infer` 只保存动作数组，不控制仿真。

全部 8 条用于本轮试训；重复轨迹组与质量信息在 `provenance/export.json` 中。
后续需要独立的新布局测试模型，不能将亮度副本或重复轨迹拆到验证集并据此声称泛化。

## 本地检查

```bash
/home/jason77/miniconda3/envs/orcalab_9_1/bin/python -m unittest \
  tests.test_tray_lerobot_export tests.test_tray_episode_recording tests.test_tray_capture_step
/home/jason77/miniconda3/envs/orcalab_9_1/bin/python -m unittest discover \
  -s examples/euler/tray_transfer/pi05_training -p test_adapter.py
HF_HUB_OFFLINE=1 /home/jason77/SY/.venv-tray-lerobot/bin/python -m unittest \
  tests.test_tray_lerobot_roundtrip
```

单卡 48 GB 的实际训练显存、loss、checkpoint 和模型闭环执行仍需云端验证。
