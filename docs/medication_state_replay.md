# 成功示范的关节状态回放

## 这次要查什么

把成功采集的数据按原时间轴重新显示到 OrcaLab，检查双臂动作、夹爪开合和数据关节顺序。当前入口是 `examples.euler.medication_sorting.replay`，不需要 ACT 或 π0.5 模型服务。

## 播放的是什么

主数据来自原始 `episode/episode.npz` 的实测 `state`：左臂 7 个关节、左夹爪 8 个关节、右臂 7 个关节、右夹爪 8 个关节，单位为弧度。按 50 Hz 播放，不做平滑、预测或插值，也不把 16 维 action 当成 30 维 state。

为让画面里的药品也对应原来的位置，脚本读取同次采集 `trace.json` 的完整场景状态。录制的 state 是执行 action 之前的状态，而 trace 的 qpos_after 是执行之后的状态，所以使用上一条 trace 的 qpos_after。每一帧的 30 个关节值和 sim_time 都必须与原始数据对齐，检查不通过就停止。

这是记录状态的可视化回放。物体随记录的位置移动，未重新进行碰撞、抓取或动力学计算，因此画面里药品进框不能当成模型推理成功，也不能证明原始 action 经控制器重放仍会成功。后续若要排查控制器，需要单独做 action 的物理执行对照。

## 怎么运行

在云端打开对应药品分类场景。不要同时运行其他仿真控制脚本；入口会获取与采集、推理共用的互斥锁。

```bash
cd /root/gpufree-data/OrcaPlayground
EP=/root/gpufree-data/datasets/g1_med_sort_v2/raw/train/episode_0000
OUT="/root/gpufree-data/training/state_replay_$(date +%Y%m%d_%H%M%S)_$$"

/opt/conda/envs/orca/bin/python -m examples.euler.medication_sorting.replay \
  --episode "$EP" \
  --output "$OUT"
```

默认原速完整播放。`episode_0000` 共 10659 帧，首尾时间差 213.16 秒。改 EP 的末尾编号即可选择其他成功示范，脚本仍会检查其成功标记及完整性。

- 只核对数据、不连接界面：增加 `--check-only`。
- 播放前 100 帧：增加 `--frames 100`，约 2 秒。
- 半速查看开合细节：增加 `--speed 0.5`。
- 停止：Ctrl+C；不恢复自主物理运行，场景保留最后显示状态。

输出目录必须是新目录，避免覆盖之前的结果。`validation.json` 记录源文件 SHA256、对齐误差及数据口径；`progress.json` 记录已发送的帧数；`result.json` 记录完成或错误。帧数指公共 render 接口已返回的请求数量，不等同于独立测量的显示器刷新帧数。

## 场景兼容检查

场景导出会重新生成部分物体关节名末尾的 UUID。检查时仅忽略这种固定格式后缀，仍比较稳定关节名、所属物体、关节类型、qpos 地址、转轴和锚点。机械臂与夹爪必须对应同一个场景结构，不会只凭维度相同就播放。

## 2026-11-05 的核对结果

- episode_0000 全部 10659 帧：30 维关节对齐最大误差 0，时间对齐最大误差 0。
- 100 帧短测：render 请求全部完成，用时约 1.98 秒。
- 负向检查：改变关节转轴或篡改一帧关节值会被拒绝；合法的导出 UUID 变化可以通过。
- 原始数据只读，未重新采集、转换、训练或上传。
- 全部 20 条成功示范、219594 帧均通过原始 state 与完整场景日志的对齐检查；报告位于云端 `/root/gpufree-data/training/state_replay_20261105/all_episode_alignment.json`。这项检查不涉及重新解码视频，也不证明训练集布局多样性或模型泛化能力。
- 整条 episode_0000 原速回放完成：10659/10659 个 render 请求完成，进程退出码 0，实测 213.1607 秒。报告：`/root/gpufree-data/training/state_replay_20261105/full_v1/result.json`。场景保持暂停，显示最后一帧。
