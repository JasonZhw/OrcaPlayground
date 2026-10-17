> 历史核查记录（2026-10-12）：下文权限阻塞和候选环境描述已经过时，不能作为当前执行命令。当前唯一解释器为 `/home/pc/ZXJ/SY/.venv/bin/python`，训练子进程使用 `.venv/pi05-deps`。依赖、基座与 GPU 小运算已通过；实时状态见 [依赖与入口说明](pi05_dependency_layer.md) 和 `runtime/reports/pi05-readiness.json`。

# π0.5 药品分类训练适配前置核查

```text
训练前置核查
├── 官方源码
│   ├── 克隆并锁定 OpenPI commit
│   ├── 显存建议与训练后端能力
│   └── Python、JAX、PyTorch 与仿真依赖冲突
├── G1 数据契约
│   ├── 实际单臂关节顺序与夹爪语义待主代理核实
│   ├── 自定义 state/action 维度填充至 32
│   └── 两路真实 RGB、任务语言和 episode 边界
├── 官方入口
│   ├── LeRobot 转换与 data transforms
│   ├── norm stats
│   └── train 与 inference
└── 只读工具与验证
    ├── 仓库、包元数据、数据和本地 checkpoint
    ├── JSON 阻塞项与非零退出码
    └── 不安装、不下载权重、不使用 GPU、不执行训练
```

## 版本和范围

核查日期：2026-10-12。已通过受控网络授权克隆 [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)，源码位于 `/home/pc/ZXJ/SY/software/openpi`，随后切换到 detached HEAD，锁定完整 commit：

```text
215abfb217dbac7d5f1273282331b9b1866c0479
```

以下源码结论均针对该 commit。工具常量 `PINNED_COMMIT` 会检查实际 HEAD 与其一致，并拒绝已修改的 tracked 工作树。未修改上游源码。

本子任务只做训练适配前置核查；OrcaLab 实时场景由主代理处理。本次仅在 `/home/pc/ZXJ/SY/pi05-venv` 创建独立候选环境，未修改固定仿真环境 `.venv`，未修改 `ZXJ_VIP`，未下载模型权重，未启动统计量计算、训练或推理。只执行了驱动/设备可见性探针；由于当前执行环境没有暴露 NVIDIA 设备节点，没有执行 GPU kernel。

## RTX 5090 / JAX GPU 冒烟结论

本次按 OpenPI 锁定依赖尝试在独立环境安装：

```bash
uv --cache-dir /tmp/pi05-jax-uv-cache venv \
  --python /usr/bin/python3.12 /home/pc/ZXJ/SY/pi05-venv --no-project
uv --cache-dir /tmp/pi05-jax-uv-cache pip install \
  --python /home/pc/ZXJ/SY/pi05-venv/bin/python 'jax[cuda12]==0.5.3'
```

环境创建实际成功，解释器为 `/usr/bin/python3.12`、Python `3.12.3`。JAX 安装实际失败：访问 `https://pypi.org/simple/jax/` 重试 3 次后出现 DNS `Temporary failure in name resolution`。失败发生在下载阶段，未发生版本解析成功、wheel 安装或 JAX 导入。

已取得的本机硬件事实：

| 探针 | 实际结果 | 含义 |
|---|---|---|
| `/proc/driver/nvidia/version` | NVIDIA open kernel module `580.159.03` | 内核驱动模块存在 |
| PCI 枚举 | `01:00.0 NVIDIA Corporation Device [10de:2b85]` | PCI 设备存在；`/proc/driver/nvidia/gpus/.../information` 标识 `NVIDIA GeForce RTX 5090` |
| `nvidia-smi` | `couldn't communicate with the NVIDIA driver` | 当前命令环境不能使用管理接口 |
| `/dev/nvidia*` | 不存在；`/dev` 被运行环境挂载为受限 tmpfs | CUDA 用户态无法获得设备节点 |
| `cuInit(0)` | 返回 `100` | CUDA 报告 `CUDA_ERROR_NO_DEVICE`，不是 JAX 版本结论 |
| JAX `jax.devices()` | 未执行 | 独立环境无 `jax/jaxlib`，且无可见 CUDA 设备 |
| 小矩阵 GPU 运算 | 未执行 | 没有可验证的 JAX GPU backend |

因此本轮对“RTX5090 能否运行 OpenPI 所需 JAX”只能给出**未验证/被环境阻塞**，不能给出 `jax.devices()` 的伪造结果。PCI 和驱动版本支持 RTX5090 的存在，但当前会话没有把 `/dev/nvidia0`、`/dev/nvidiactl`、`/dev/nvidia-uvm` 暴露给进程；这必须先由主机/容器运行方修复。自动审批服务拒绝了读取真实主机 `nvidia-smi` 的 escalation 请求，返回 `502 Bad Gateway`，没有绕过该限制。

本次没有尝试兼容小升级。OpenPI `uv.lock` 锁定 `jax/jaxlib/jax-cuda12-plugin/jax-cuda12-pjrt` 为 `0.5.3`；若设备节点和网络恢复，优先复测该精确版本。只有精确版本在真实设备可见时因 Blackwell/XLA 错误失败，才应记录错误后尝试官方可用的最小 JAX/JAX CUDA12 升级，并把升级版本、lock 偏差、CUDA plugin 版本和 smoke 输出写入本节。不能把“可考虑升级”写成已验证兼容。

完整 GPU 验证应在真实主机设备可见的 `/home/pc/ZXJ/SY/pi05-venv` 中执行，且只做小矩阵：

```bash
/home/pc/ZXJ/SY/pi05-venv/bin/python - <<'PY'
import jax
import jax.numpy as jnp

print("jax", jax.__version__)
print("jax_devices", jax.devices())
device = jax.devices("gpu")[0]
a = jax.device_put(jnp.arange(16, dtype=jnp.float32).reshape(4, 4), device)
b = jax.device_put(jnp.eye(4, dtype=jnp.float32), device)
c = (a @ b).block_until_ready()
print("device", device)
print("matrix_shape", c.shape)
print("matrix_sum", float(c.sum()))
PY
```

预期可核对值为 `matrix_shape=(4, 4)`、`matrix_sum=120.0`；在本轮没有执行，因此不把预期值当作实际结果。

## 显存与 LoRA 能力

依据 [OpenPI README](/home/pc/ZXJ/SY/software/openpi/README.md:22)，官方单 GPU 显存估算如下：

| 模式 | 官方要求 | 官方示例 |
|---|---|---|
| 推理 | `>8 GB` | RTX 4090 |
| LoRA 微调 | `>22.5 GB` | RTX 4090 |
| 全参微调 | `>70 GB` | A100 80GB / H100 |

这是仓库级估算，没有给出本任务的 π0.5/G1/双视角/batch 实测显存。现有主计划记录 RTX 5090 32GB；本次未重新查询硬件。按该容量，LoRA 可作为后续核验方向，全参不满足官方单卡估算。实际空闲显存、OrcaLab 同时占用以及 Blackwell 对锁定 JAX/CUDA 组合的兼容性均未验证，不能据此宣布可训练。

[PyTorch 支持说明](/home/pc/ZXJ/SY/software/openpi/README.md:190) 明确不支持 LoRA、FSDP、混合精度训练和 EMA。JAX 的 [Pi0Config.get_freeze_filter](/home/pc/ZXJ/SY/software/openpi/src/openpi/models/pi0_config.py:88)、[Gemma LoRA 实现](/home/pc/ZXJ/SY/software/openpi/src/openpi/models/gemma.py:80) 与 [训练参数过滤](/home/pc/ZXJ/SY/software/openpi/scripts/train.py:156) 提供 LoRA 机制。

现成低显存配置 `pi0_libero_low_mem_finetune` 属于 π0，见 [config.py](/home/pc/ZXJ/SY/software/openpi/src/openpi/training/config.py:678)。本 commit 的注册表没有现成的 π0.5 LoRA/G1 配置。后续需用 `Pi0Config(pi05=True, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")`、匹配的 `get_freeze_filter()`、`ema_decay=None` 和 `pi05_base/params` 构建并验证自定义 JAX 配置；此组合是根据源码提出的适配方向，本次没有添加配置或实测。

官方 `pi05_libero` 是全参示例，`batch_size=256`、`action_horizon=10`、`discrete_state_input=False`，见 [config.py](/home/pc/ZXJ/SY/software/openpi/src/openpi/training/config.py:743)。不能将它直接作为 32GB G1 LoRA 配置。

## Python、PyTorch、JAX 和仿真依赖

实际声明依据 [pyproject.toml](/home/pc/ZXJ/SY/software/openpi/pyproject.toml:6) 与 [uv.lock](/home/pc/ZXJ/SY/software/openpi/uv.lock:1)：

| 项目 | OpenPI 声明/锁定 | 当前 `.venv` 包元数据 | 结论 |
|---|---|---|---|
| Python | `>=3.11`；lock 包含 3.12 分支 | `3.12.14` | 满足声明；没有运行训练栈 |
| PyTorch | `torch==2.7.1` | `2.14.1` | 确定版本冲突 |
| NumPy | `>=1.22.4,<2.0.0`；lock 为 `1.26.4` | `2.2.6` | 确定版本冲突 |
| JAX | `jax[cuda12]==0.5.3`；jaxlib/plugin/pjrt 同为 `0.5.3` | 独立 `pi05-venv` 安装受 DNS 阻塞，仍未安装 | 阻塞；设备节点、JAX 0.5.3 CUDA12 和 Blackwell 未验证 |
| Flax | `0.10.2` | 未安装 | 阻塞 |
| Orbax | `orbax-checkpoint==0.11.13` | 未安装 | 阻塞 |
| transformers | `4.53.2` | 未安装 | 阻塞；PyTorch 路径还需官方源码补丁 |
| OpenPI / client | 主仓库和 workspace package | 均未安装 | 有源码不等于已安装 |
| LeRobot | Git rev `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`，lock 中版本 `0.1.0` | 未安装 | 后续必须使用该 rev，不能默认兼容当前最新版 |

依赖覆盖还包括 `ml-dtypes==0.4.1`、`tensorstore==0.1.74`，见 [pyproject.toml](/home/pc/ZXJ/SY/software/openpi/pyproject.toml:65)。RLDS 可选组固定 `tensorflow-cpu==2.15.0`，源码注明须 Python 3.11，见 [pyproject.toml](/home/pc/ZXJ/SY/software/openpi/pyproject.toml:57)；这不等于整个 OpenPI 强制 Python 3.11。自定义 LeRobot 数据不必照搬依赖 TensorFlow 的 LIBERO 原始数据转换器。

只读查询已安装 distribution 的 Requires-Dist，发现 `orca-gym 26.9.1` 明确要求 `numpy==2.2.6`，当前 `opencv-python 5.0.0.93` 在 Python >=3.9 下要求 `numpy>=2`。它们与 OpenPI 的 NumPy <2 没有交集。现有 PyTorch distribution 的依赖还有 `cuda-toolkit`、`nvidia-cudnn-cu13` 等 CUDA13 包，而 OpenPI 锁定的是 CUDA12 JAX/PyTorch 依赖系列。这些约束不能通过直接向项目 `.venv` 执行 `uv sync` 解决。

当前系统 `/etc/os-release` 为 Ubuntu 24.04.4 LTS；README 只报告测试 Ubuntu 22.04 并声明不支持其他系统，见 [README](/home/pc/ZXJ/SY/software/openpi/README.md:32)。后续兼容性验证由主代理安排，不在本次安装或改动环境。PyTorch 文档中的 transformers 补丁路径 `python3.11` 只是示例，见 [README](/home/pc/ZXJ/SY/software/openpi/README.md:200)；不能原样覆盖当前 Python 3.12 环境。

## G1 单臂与夹爪的维度契约

[Pi0Config](/home/pc/ZXJ/SY/software/openpi/src/openpi/models/pi0_config.py:18) 默认 `action_dim=32`、`action_horizon=50`；π0.5 默认 `max_token_len=200`、`discrete_state_input=True`。这表示预训练模型的输入/输出宽度，不代表 G1 有 32 个可控关节。π0.5 DROID 微调配置也特意保留 32 维，见 [config.py](/home/pc/ZXJ/SY/software/openpi/src/openpi/training/config.py:899)。[模型动作投影](/home/pc/ZXJ/SY/software/openpi/src/openpi/models/pi0.py:92) 与 action_dim 绑定，不应直接改为 8 并假设预训练权重仍兼容。

本任务拟采用同一顺序的单臂关节位置与夹爪状态/目标：

```text
D = 实时确认的单臂关节数 N + 实时确认的夹爪控制维度 G
state(t) = [q_1(t), ..., q_N(t), gripper_state(t)]
action(t) = [q_target_1(t), ..., q_target_N(t), gripper_target(t)]
原始 state: (D,)；模型动作 chunk: (H, D)
```

若主代理确认 N=7、夹爪经验证可合并为 G=1，才可采用 D=8；这只是条件示例，当前 G1 数量、名字和夹爪传动关系尚未确认。不能直接使用全身 qpos 或猜测夹爪闭合方向。关节以 rad 表示，夹爪归一化区间、开闭方向、控制频率和动作是位置/速度/增量都需要写入采集契约；DROID 的关节速度语义不能直接复用为 G1 位置目标。

[PadStatesAndActions](/home/pc/ZXJ/SY/software/openpi/src/openpi/transforms.py:327) 在 normalization 后将原始 D 维补至 32；推理输出需反归一化、恢复选定动作语义，然后只保留前 D 维并按真实执行器顺序发送。若 D>32，零填充无法适配。state/action 在 OpenPI 框架中并不普遍要求原始维度相等（LIBERO 是 8/7）；本工具要求同维是本任务拟定的“单臂关节+夹爪位置”契约。

[LiberoOutputs](/home/pc/ZXJ/SY/software/openpi/src/openpi/policies/libero_policy.py:86) 固定截取 7 维，[DroidOutputs](/home/pc/ZXJ/SY/software/openpi/src/openpi/policies/droid_policy.py:77) 固定截取 8 维；后续 G1 adapter 必须用实际 D。对绝对位置目标是否启用 [DeltaActions/AbsoluteActions](/home/pc/ZXJ/SY/software/openpi/src/openpi/transforms.py:203)，必须与训练/推理一致；夹爪通常保留绝对值，不能重复应用 delta transform。

核查工具读取 state/action shape 和 names，只将名称视为 `metadata_declared`，始终报告 `runtime_verified=false`，不会声称已验收 G1 关节映射。

## RGB 双视角与 data transforms

[Pi0Config.inputs_spec](/home/pc/ZXJ/SY/software/openpi/src/openpi/models/pi0_config.py:64) 列出三槽位：`base_0_rgb`、`left_wrist_0_rgb`、`right_wrist_0_rgb`。[DroidInputs](/home/pc/ZXJ/SY/software/openpi/src/openpi/policies/droid_policy.py:42) 与 [LiberoInputs](/home/pc/ZXJ/SY/software/openpi/src/openpi/policies/libero_policy.py:42) 展示了两路真实视角加一个零填充/mask=false 的方式。后续 G1 需固定外部/头部视角与实际单臂腕部视角，并明确 wrist 对应槽位；主代理仍需验证两路 RGB 的实际采集、颜色通道、时间同步与覆盖范围。

LeRobot 解码后可能是 float32 CHW，在线相机可能是 uint8 HWC。官方适配器会转换布局/范围，[ModelTransformFactory](/home/pc/ZXJ/SY/software/openpi/src/openpi/training/config.py:106) 再以 `ResizeImages(224,224)` 保持比例填充。readiness 只检查 metadata 中至少两个 image/video 特征、三通道 shape 以及 base/wrist 名称线索。它不解码图像、不核实帧同步，不把模型内部零填充视为第二路真实采集图像。

[训练数据变换顺序](/home/pc/ZXJ/SY/software/openpi/src/openpi/training/data_loader.py:172) 为：

```text
LeRobot task → prompt
→ repack_transforms（原始字段名映射）
→ data_transforms（机器人/图像/动作语义适配）
→ Normalize（π0.5 默认 quantile normalization）
→ model_transforms（resize、语言与 state token、32维填充）
```

[LeRobotLiberoDataConfig](/home/pc/ZXJ/SY/software/openpi/src/openpi/training/config.py:282) 说明 repack 通常只用于训练，机器人 data transforms 同时用于训练和推理。G1 需要独立 adapter/config；本次未修改这些上游文件，也没有注册虚构的 `pi05_medication_sorting` 配置。

## LeRobot、norm stats 与官方入口

[OpenPI LeRobot 转换示例](/home/pc/ZXJ/SY/software/openpi/examples/libero/convert_libero_data_to_lerobot.py:43) 使用 `LeRobotDataset.create` 定义 fps/features，`add_frame` 写图像、state、actions、task，并通过 `save_episode` 保留 episode 边界。字段名不是统一固定值，取决于 repack 配置；当前工具接受 `observation.state`/`state` 和 `action`/`actions`，但额外要求逐维名称以方便 G1 核查。LeRobot feature 的普通 names 标签不一定逐维列出关节，本工具的名称要求是项目适配约束。

[数据加载器](/home/pc/ZXJ/SY/software/openpi/src/openpi/training/data_loader.py:130) 按 `repo_id` 使用锁定的 `lerobot.common.datasets.lerobot_dataset` API，并通过 `[t/fps for t in range(action_horizon)]` 生成动作 chunk。它没有把任意 `--data` 路径直接传给 LeRobot root；官方转换器使用 `HF_LEROBOT_HOME / REPO_NAME`，见 [转换器](/home/pc/ZXJ/SY/software/openpi/examples/libero/convert_libero_data_to_lerobot.py:37)。后续导出目录/cache 与自定义 repo_id 必须配套配置，readiness 的 `--data` 只指定核查路径。

工具核查 `meta/info.json`、fps、数值 feature shape、names、两路 RGB 和 `data/**/*.parquet` 的非空文件；语言元数据兼容候选 `meta/tasks.jsonl`、`tasks.parquet`、`tasks.json` 或 prompt feature。它不读取 Parquet 内容、不检查 episode 完整性或任务语义。锁定 LeRobot commit 的原始文本进一步核查受审批服务 502 阻塞，未执行；不能据此声称已确认具体 LeRobot schema/version 或最新版兼容性。正式导出必须对该 Git rev 实测。

[norm stats 官方说明](/home/pc/ZXJ/SY/software/openpi/docs/norm_stats.md:1) 要求 state/action 的统计量随训练与推理使用同一套定义。G1 不应因为维度同为 8 就直接复用 DROID/Franka statistics。[compute_norm_stats.py](/home/pc/ZXJ/SY/software/openpi/scripts/compute_norm_stats.py:89) 在 repack 和机器人 transforms 后计算 `state`、`actions`，将 `norm_stats.json` 写到 `config.assets_dirs / data_config.repo_id`。[归一化保存格式](/home/pc/ZXJ/SY/software/openpi/src/openpi/shared/normalize.py:120) 包含外层 `norm_stats` 与 mean/std/q01/q99；π0.5 默认使用分位数。

[checkpoint 保存](/home/pc/ZXJ/SY/software/openpi/src/openpi/training/checkpoints.py:65) 将统计量放入 `assets/<asset_id>/norm_stats.json`。[create_trained_policy](/home/pc/ZXJ/SY/software/openpi/src/openpi/policies/policy_config.py:45) 自动识别 `model.safetensors` 或 JAX `params/`，再从 checkpoint assets 读取统计量。readiness 要求显式指定本地 checkpoint 根目录，检查非空权重文件/参数目录及 stats JSON；不会恢复参数或验证权重完整性、模型类型、机器人身份及统计量来源。基础权重与 G1 norm stats 尚需后续分别准备；仅有基础 checkpoint 不代表 G1 已适配。

| 操作 | 官方入口与参数示例 | 本次执行 |
|---|---|---|
| 计算统计量 | `scripts/compute_norm_stats.py --config-name pi05_libero` | 未执行 |
| JAX 训练 | `scripts/train.py pi05_libero --exp-name=my_experiment` | 未执行 |
| PyTorch 训练 | `scripts/train_pytorch.py <config_name> --exp_name <run_name>` | 未执行 |
| JAX→PyTorch 权重转换 | `examples/convert_jax_model_to_pytorch.py --checkpoint_dir ... --config_name ... --output_path ...` | 未执行 |
| 推理服务 | `scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir=...` | 未执行 |

这些入口来自 [README](/home/pc/ZXJ/SY/software/openpi/README.md:151) 和 [PyTorch 说明](/home/pc/ZXJ/SY/software/openpi/README.md:254)。示例属于上游 LIBERO/DROID 等场景，不是已完成的 G1 药品任务命令。PyTorch 训练脚本仍导入 JAX 并复用公共数据管线，见 [train_pytorch.py](/home/pc/ZXJ/SY/software/openpi/scripts/train_pytorch.py:34)，不能用它避开当前 JAX 依赖核查。任何调用 OpenPI `maybe_download`、`uv run` 或模型入口都不属于本工具行为。

## 工具用法与实际验证

```bash
PYTHONDONTWRITEBYTECODE=1 ./.venv/bin/python \
  tools/medication_sorting/check_training_readiness.py --pretty
```

可用参数：`--project-root`、`--repo`、`--data`、`--checkpoint`、`--venv`、`--pretty`。默认项目根目录从工具路径推导，固定环境默认为该根目录的 `.venv`。正常核查只向 stdout 输出 JSON；发现 blockers 时退出 1，通过这些本地 metadata 检查时退出 0。`ready` 只对应 `verification_scope=local_metadata_only`，不是训练/依赖导入/数据质量或仿真验收通过。

实际默认执行结果：

```json
{
  "ready": false,
  "repository_commit": "215abfb217dbac7d5f1273282331b9b1866c0479",
  "blocker_count": 11,
  "execution": {
    "training_started": false,
    "weights_downloaded": false,
    "gpu_touched": false,
    "data_written": false
  }
}
```

上面是完整 JSON 的摘要。实际 blockers：7 项关键包缺失、torch/numpy 两项版本冲突、`data/medication_sorting/meta/info.json` 缺失，以及本地 `checkpoints` 缺失。当前数据目录只有场景与报告，未产生已确认的 LeRobot episode。

测试命令：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  ./.venv/bin/python -m pytest -q -p no:cacheprovider \
  tools/medication_sorting/tests/test_training_readiness.py
```

实际结果为 `21 passed`。禁用插件自动加载是因为本机 ROS `launch_testing` 插件因缺少 `lark` 在测试收集前报错；没有因此安装或改动任何包。测试只在 pytest 的 `/tmp` 临时目录使用明确的测试 fixture，未生成真实训练数据或模型；fixture 中的非空文件不代表实际权重有效。

覆盖仓库缺失/commit 不可解析、数据元数据缺失/损坏、维度不一致/非法/超32、缺少 RGB 双视角、灰度图不能算 RGB、只有名称不能证明实时映射、空记录、空参数目录、stats 损坏、远程权重不会下载、环境版本冲突、JSON 与退出码。初始缺模块失败及新增关键失败分支均实际观察后再实现。

## 交给主代理的阻塞事项

1. 现有 `.venv` 的 OrcaGym NumPy2 要求与 OpenPI NumPy<2 确定冲突。需主代理确认依赖迁移/隔离方式，不能直接执行安装或另建未经授权的项目环境。
2. G1 单臂关节、夹爪、动作语义和相机映射仍须实时验收；N/G/D 尚未锁定。
3. 需要真实双视角图像、state/action 时间同步、语言任务、episode 成功与划分记录，并按锁定 LeRobot rev 验证导出与读取。
4. G1 data transforms、训练配置与 norm stats 尚未实现；π0.5 LoRA 自定义配置仅是源码支持的候选方向。
5. 本地基础 checkpoint 尚未准备，下载及长训仍不执行；RTX5090/JAX0.5.3/CUDA12/Ubuntu24.04 组合本轮未完成 GPU smoke。
6. `/home/pc/ZXJ/SY/pi05-venv` 已创建但为空；JAX 下载因 DNS 失败，不能提供真实 `jax.devices()` 或小矩阵结果。
7. 当前执行环境缺少 `/dev/nvidia*`，`cuInit(0)=100`；需真实主机/容器修复设备透传后重跑精确 JAX 0.5.3。
8. 主机 `nvidia-smi` escalation 被自动审批服务以 `502 Bad Gateway` 拒绝，属于审批失败，不是模型源码不存在，也不是依赖兼容性的结论。本次没有绕过审批。

本次交付是可复核的前置检查与失败说明，没有训练成功、抓取成功、有效训练数据或模型可用的结论。
