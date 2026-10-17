# 药品自动抓取、采集与 π0.5

> 2026-10-17交接说明：本文件保留了开发期间的阶段记录，部分“正在采集”等状态已过时。新电脑复现请先阅读[药品分类项目开发记录与复现手册](../../docs/medication_sorting/药品分类项目开发记录与复现手册.md)，以实际数据和检查点记录为准。

已建立单药盒基线；当前扩展为两个药盒放紫筐、两个药瓶放蓝筐。
四物品处于物理试抓和小批试采阶段，不能直接复用单药盒结果作为四物品验收。

## 四药品160条与20,000步新实验

用户已确认每种40条合格随机示范，按32/8分层训练验证，最终每种10次新随机摆放
测试。范围为目标初始位置每轴正负2cm、转角正负15度、保持直立；两瓶为70%尺寸。
新专家配置`randomized_tabletop_70`保持旧配置不变。采集尚在进行，不能宣称已有
160条数据或完成多物品训练。正式状态目录为
`data/medication_sorting/multi_object/randomized160-20261014/production`。
长任务入口`run_randomized_experiment.py`，先验收足量数据才训练，错误会停止并留档。
只有采集阶段支持断点补采；后续阶段中断需要先核对已有输出，服务不会自动重试。
2026-10-14夜间另启`sy-medication-night-watch.timer`，每次检查结束后60秒复查，
仅安全恢复采集，至10月15日08:00结束守护；实时状态保存在正式目录`night_watchdog.json`。
完整说明见`docs/medication_sorting/randomized160_experiment.md`。

## 缩小药瓶独立实验（2026-10-14）

用户授权后，创建了 `data/medication_sorting/scenes/drag_classfication_bottles70.json`。
只将两个药瓶比例乘以0.7，其他对象、位置和原场景不变；来源及SHA256保存在同名
`manifest.json`。大瓶直径9→6.3cm，小瓶8→5.6cm，质量也随资产缩放改变为约
343g、256g，不能将其称为原尺寸药瓶任务通过。

独立物理试验 `multi_object/bottles70-all-four-physics-20261014/summary.json`
中四种物品各一次抓取落筐成功。此为每物品重新初始化的试验，尚非连续四物品整理。
专用 `scaled_bottles_70` 配置采用较低的搬运高度、大瓶近筐底释放；旧默认参数保留。

恢复已生成的实验场景并做新一批采集：

```bash
cd /home/pc/ZXJ/SY
.venv/bin/python tools/medication_sorting/prepare_scene.py \
  --layout data/medication_sorting/scenes/drag_classfication_bottles70.json --record-seconds 0
.venv/bin/python -m tools.medication_sorting.run_four_objects --rgb \
  --profile scaled_bottles_70 \
  --layout data/medication_sorting/scenes/drag_classfication_bottles70.json \
  --output data/medication_sorting/multi_object/new-bottles70-rgb-pilot
```

缩小配置必须显式指定相符的实验场景，执行前检查实际两瓶碰撞直径；报告和采集
元数据记录 `expert_profile`，离线验收检查两者一致。不能使用旧单药盒模型直接
宣称四物品推理成功；试采和后续训练数据均与旧数据分开。
细节见 `docs/plans/2026-10-14-缩小药瓶抓取试验.md`。

试采最终结果：5次录制中4次合格，共392帧，四种物品各1条。原会话的41帧大瓶
记录因连续空白图像被拒收；重启引擎后另存98帧合格补采，没有修补失败回合。
只读复核汇总：`data/medication_sorting/multi_object/bottles70-reviewed-20261014.json`。
长批次仍有间歇黑图问题，未完整修复；4条试采不足以开始可靠的多物品训练。

## 四物品试验入口（2026-10-14）

先按下文恢复 `drag_classfication_capture.json`，再运行：

```bash
cd /home/pc/ZXJ/SY
.venv/bin/python -m tools.medication_sorting.run_four_objects --rgb \
  --output data/medication_sorting/multi_object/new-four-object-pilot
```

每个物品独立重置和采集一回合，目录包含双 RGB、8 维 state/action、
逐帧证据和真实执行报告；失败回合仍保留，但 `success:false`，禁止训练。
`summary.json` 按物品列出真实成功与质量合格条数；只有四种目标都有合格
数据时入口才退出 0。`--targets` 可选择子集，但不能据此声称四物品通过。
不加 `--rgb` 只做物理试抓，不是视觉训练数据。

连续整理的入口：

```bash
.venv/bin/python -m tools.medication_sorting.run_four_objects --continuous \
  --output data/medication_sorting/multi_object/new-continuous-physics-pilot
```

此模式仅首物品重置，后续不 reset、不改物体 qpos；同筐使用分开的落点，
失败立即停止。最终重新检查四物品是否仍在正确筐中。
`continuous_sorting_success` 与“独立抓取四个均成功”分开统计。
当前药瓶参数尚未验证成功，不应批量采集或启动四物品 π0.5 训练。
这批原尺寸试验未缩小药瓶、改变物体质量或加吸附约束；原始 Layout 不改动。

本轮实测：`multi_object/four-object-rgb-pilot-20261014/reviewed_summary.json`
记录四种物品各一次录制，共282帧；两个药盒的199帧通过验收，两个药瓶的
83帧因抬升失败被拒收。连续模式第二个药盒搬运逆解超限，尚未通过。
详细计划及阻塞点见 `docs/plans/2026-10-14-四物品抓取与采集.md`。

以下是已有单药盒基线的操作与历史记录。

原始场景保留。当前实验副本使用 `drag_classfication_capture.json`：G1 平移
[.14,-.18,0]，紫筐放大 20%。专家使用 IK 与真实 MuJoCo 接触，不在回合中
瞬移药盒、不加吸附约束；当前没有使用 CuRobo 规划。

## 当前证据

- 20 次固定初态专家测试成功：`data/medication_sorting/experiments/repeatability20-5hz/summary.json`。
- 首个合格视觉回合：`data/medication_sorting/raw/20261013-121258/episode_0000`，96 帧，5 Hz。
- 随机初态专家采集已完成：`data/medication_sorting/raw/20261013-122336`，25/25 成功；连同早期一个合格回合共 26 回合。
- 实际数据检查通过：`runtime/reports/pi05-final-data-preflight.json`；21 个训练回合（2016 帧）和 5 个验证回合（480 帧）。
- 1500 步 π0.5 微调及 `1499` 保存完成；真实推理 seeds 413..417 成功 1/5，完整记录于 `runtime/reports/pi05-1499-heldout-summary.json`。
- 新实验 `g1_medication_20261013_refine` 已完成 6000 步及 `5999` 保存，新的 seeds 513..517 真实推理成功 3/5（60%）。见 `runtime/reports/pi05-5999-heldout-summary-verified.json`，尚不足以通过稳定任务验收。
- 6000 步版本在第二批 seeds 613..617 成功 1/5，613 完成落筐；614..616 未稳定抓起，617 抬起后放置失败。两批合计 4/10，见 `runtime/reports/pi05-5999-final613-verified.json`；稳定验收仍未通过。
- 分批保存后的 12000 步模型 `g1_medication_20261013_refine12k/11999` 已完整提交并恢复到服务；新 seeds 713..717 成功2/5，仍不稳定。检查点核验：`runtime/reports/pi05-11999-checkpoint-verification.json`；实测结果：`runtime/reports/pi05-11999-first713-verified.json`。

## 1. 恢复实验场景

已有 OrcaLab 窗口时直接加载。采集或策略执行期间不要重复执行恢复命令。

```bash
cd /home/pc/ZXJ/SY
.venv/bin/python tools/medication_sorting/prepare_scene.py \
  --layout data/medication_sorting/scenes/drag_classfication_capture.json \
  --record-seconds 0
```

本机 MCP 为 `http://127.0.0.1:12345/mcp`，gRPC 为 `127.0.0.1:50051`。
恢复入口先备份当前编辑场景，再加载、核对并启动目标场景。
没有窗口时，先按 `tools/orca/environment.sh` 设置环境，从 Playground 目录
启动 `.venv/bin/orcalab --verbose --lang zh_CN`，随后执行恢复入口。
`start_orcalab.sh` 默认加载原始浮动基座场景，不能直接把它当成当前采集场景。

## 2. 自动采集

```bash
.venv/bin/python -m tools.medication_sorting.collect_episodes \
  --episodes 25 --randomize --seed 202610131 --control-hz 5
```

采集前观测，随后执行动作。每行包含两路 RGB、7 个右臂关节与夹爪的
8 维状态/绝对目标动作、仿真时间、阶段和英文任务文本。RGB 为 224×224，
固定 gamma=.45；物理步长 .001 秒，每条动作推进 200 步。

当前两路图像来自 OrcaLab 官方 MCP 原生视口渲染：固定俯视和虚拟手端跟随。
手端视角不是物理安装的腕部传感器。原生 head_cam/camera_right 黑图问题尚未
解决。采集与推理共用相机和图像处理；两次取图之间不步进。

偶发黑图最多保持原状态重绘三次，拒收图像保存为 `rejected_*.png`。
持续黑图、中断或物理失败回合保留证据，但不进入训练。

补充修正接近位置的示范时，可加 `--approach-jitter .02`：仅盒上方的接近航点每轴偏移最多2cm，下探恢复实际药盒中心。当前先在 `raw/20261013-recovery-pilot` 做3回合试验；扰动不修改物体状态，物理失败或图像验收失败的数据仍被排除。

## 3. 数据验收与转换

当前正式划分为 `data/medication_sorting/splits/g1-reviewed-20261013`，21 个训练回合、5 个验证回合。内容完全相同的副本归为同一组，不按相邻帧拆分。正式数据集已建立；下面是建立新数据集时的流程，现有目标目录不要重复转换。

```bash
.venv/bin/python -m tools.medication_sorting.verify_episodes \
  --input data/medication_sorting/raw/20261013-122336 \
  --summary > runtime/reports/rechecked-collection-quality.json

export HF_LEROBOT_HOME=/home/pc/ZXJ/SY/data/medication_sorting/lerobot
export OPENPI_DATA_HOME=/home/pc/ZXJ/SY/runtime/cache/openpi
tools/medication_sorting/pi05_python.sh --cpu \
  -m tools.medication_sorting.convert_episodes \
  --input data/medication_sorting/splits/g1-reviewed-20261013/train \
  --output "$HF_LEROBOT_HOME/local/g1_medication_train" \
  --repo-id local/g1_medication_train --fps 5
tools/medication_sorting/pi05_python.sh --cpu \
  -m tools.medication_sorting.pi05_adapter --repo-id local/g1_medication_train norm-stats
```

统计量只用训练回合。前 7 维动作变为相对当前状态的关节增量，夹爪保留
0=张开、1=闭合。模型内部填充到 32 维，部署反变换恢复实际 8 维动作。

## 4. 训练

先保存并关闭 OrcaLab，释放 GPU，再执行真实训练。以下 1 步用于检查权重
加载、反向传播和 checkpoint 保存；它不能证明模型已经学会抓取。

```bash
tools/medication_sorting/pi05_python.sh \
  -m tools.medication_sorting.pi05_adapter --repo-id local/g1_medication_train \
  --batch-size 1 --action-horizon 10 train -- \
  --exp-name=g1_smoke \
  --weight-loader.params-path=/home/pc/ZXJ/SY/data/medication_sorting/pretrained/pi05_base/params \
  --num-train-steps=1 --save-interval=1 --log-interval=1
```

当前已有真实检查点：`data/medication_sorting/checkpoints/pi05_g1_medication_sorting_lora/g1_medication_20261013_refine/5999`。正式微调使用独立实验名和实际确定的训练步数。环境继续使用同一
`.venv/bin/python`；wrapper 仅为训练子进程启用 `.venv/pi05-deps`。
不修改 OpenPI 上游或 `ZXJ_VIP`。版本、兼容差异及命令说明见
`docs/medication_sorting/pi05_dependency_layer.md`。

## 5. 部署与仿真推理

服务必须加载真实微调 checkpoint，`--policy.dir` 指向同时包含 `params/`
和 `assets/` 的编号目录。训练与服务的 action horizon 保持 10。

```bash
tools/medication_sorting/pi05_python.sh \
  -m tools.medication_sorting.pi05_adapter --repo-id local/g1_medication_train \
  --action-horizon 10 serve -- --policy.dir=/home/pc/ZXJ/SY/data/medication_sorting/checkpoints/pi05_g1_medication_sorting_lora/g1_medication_20261013_refine/5999 --port=8000
```

恢复实验场景后，另开终端执行策略。评估驱动对本地 WebSocket 子进程屏蔽代理，逐回合保留报告、照片及动作日志：

```bash
.venv/bin/python tools/medication_sorting/evaluate_policy.py \
  --checkpoint-reference /home/pc/ZXJ/SY/data/medication_sorting/checkpoints/pi05_g1_medication_sorting_lora/g1_medication_20261013_refine/5999 \
  --training-manifest data/medication_sorting/splits/g1-reviewed-20261013/split_manifest.json \
  --seed-start 613 --episodes 5 --control-hz 5 --execute-chunk 4 --steps 150 \
  --output /absolute/new_evaluation
```

推理只读取双 RGB、机器人状态与任务提示，由模型生成动作；物体真值用于
验收与日志，不用于生成策略动作。使用未训练的初始 seed 测试真实抬升、
释放、落筐和稳定。返回动作、训练 loss 下降或服务启动均不等于任务成功。

## 回归检查

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest \
  tools/medication_sorting/tests tests -q
```

## 训练保存内存

主机为 30GiB RAM、8GiB swap。3GB 写读缓冲限额曾支持完整 1499 和 5999 保存，但 12000 步最终保存仍被 Linux OOM killer 终止：Orbax 在写盘前同时复制所有设备数组，缓冲限额不能约束这一峰值。

当前适配器通过局部 type handler 逐数组复制到主机、等待写盘、释放缓冲，params 和优化器共享复制锁。沿用 Orbax 格式与 sharding，不修改 OpenPI 或全局 handler。298 项回归、真实 CPU 复合检查点恢复、本轮完整 11999 保存和服务恢复已通过。本次进程 RSS 峰值约19GiB，最低主机可用内存约5GiB，仍应保留内存余量。训练时停止本项目仿真和模型服务。

## 当前实验范围

已跑通自动专家采集 → LeRobot 数据 → π0.5 LoRA 训练 → WebSocket 策略 → 真实 MuJoCo 抓取落筐。数据仅覆盖固定基座、单药盒、毫米级初态扰动和实验放大紫筐；两路相机为实际原生渲染的固定俯视及虚拟手端跟随。6000 步模型两批共 4/10，不能推广到原尺寸筐、药瓶、多药品分类或真实机器人。

## 已执行的 12000 步继续微调

首次 12000 步训练在最终保存时 OOM，没有可用新模型。临时目录已保留为 `11999.orbax-checkpoint-tmp-OOM`。2026-10-13 修复分批保存后，从保留的完整 `5999` 恢复优化器继续到 12000，峰值学习率为 1e-5。当前训练日志为 `runtime/logs/pi05-train-12000-streaming.log`，命令如下：

```bash
HF_LEROBOT_HOME=/home/pc/ZXJ/SY/data/medication_sorting/lerobot \
OPENPI_DATA_HOME=/home/pc/ZXJ/SY/runtime/cache/openpi \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
PYTHONUNBUFFERED=1 tools/medication_sorting/pi05_python.sh \
  -m tools.medication_sorting.pi05_adapter --repo-id local/g1_medication_train \
  --batch-size 1 --action-horizon 10 train -- \
  --exp-name=g1_medication_20261013_refine12k --resume \
  --num-train-steps=12000 --save-interval=12000 --log-interval=200 \
  --lr-schedule.warmup-steps=100 --lr-schedule.peak-lr=1e-5 \
  --lr-schedule.decay-steps=12000 --lr-schedule.decay-lr=3e-6
```

回合 reset 与示范一致，但控制策略只接收机器人状态、双 RGB 和任务提示；物体位置用于日志及验收，不生成模型动作。诊断 `execute_chunk=1` 的 514..516 三回合均失败，未作为正式执行配置。
