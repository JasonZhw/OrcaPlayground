# 来源与许可证

- pi0.5背景及图像工具：Physical-Intelligence/openpi。
  `third_party/openpi/image_tools.py` 是固定提交的原文件，
  Apache-2.0许可证位于同目录 `LICENSE`。
- 训练实现：PyPI LeRobot0.6.1，仓库 `huggingface/lerobot`；
  本项目不复制或修改其模型/训练器源码。
- cuRobo：NVlabs/curobo，固定提交由安装脚本获取。
  上游 `LICENSE`、`LICENSE_ASSETS` 保存在 `third_party/curobo/`。
- 原始示范：`xiaojia666/g1-blue-intel-single-target-200`。
- 已训练权重：`Harrysunshine/g1-pi05-v9-single-target-200`。
- 初始模型和tokenizer：`lerobot/pi05_base`、
  `google/paligemma-3b-pt-224`，遵循各自模型仓库的访问和使用条款。
- 场景XML、标定、夹爪网格来自本V9采集工程；可视资源标识列在布局中。
  OrcaLab/OrcaStudio包的分发与使用按平台及对应资产条款执行，
  本仓库的代码许可不替代第三方资产许可。

固定revision、源文件哈希和运行环境分别记录在 `configs/` 与 `reference/training/`。
