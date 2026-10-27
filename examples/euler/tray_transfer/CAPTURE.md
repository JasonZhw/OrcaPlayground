# 当前包装盒任务：双相机示范采集

当前任务是把桌面上浅蓝色且最小的盒子放进左侧木托盘。机器人先放下左臂，再录制右臂抓取、搬运、松爪撤离和回位。只采集 `head_cam` 与 `camera_right`。

## 已采集数据

当前共保留 8 条用户验收示范、26,617 帧，来自 5 个布局批次；其中 7 条包含右臂回位。已按 [转换清单](lerobot_manifest_v1.json) 整理为 LeRobot v2.1，结果与云端训练包位置见 [数据转换说明](LEROBOT.md)。

以下启动命令对应最新一批布局，不是任意新布局的通用抓取计划。

## 最新批次

本批目录：`/home/jason77/SY/datasets/tray_transfer/layout_20261027_232508`。
已录制1条，3395组同步样本、67.90秒，包含右臂回位。抓姿相对竖直倾斜5°，重新求解当前布局的IK；抓取前闭合d=.97、对齐a=±.28/d=.90、搬运锁存d=1.45，释放高度12mm。

## 成功标准

本计划配置 `placement.acceptance="released_inside_tray"`。物体完整XY边界进入托盘（允许1mm接触容差）、底部接近托盘底面、夹爪松开后即可开始回位；不要求物体摆正、远离框边或通过静止速度/底面独立支撑检查。结束时确认右臂回到默认位置且盒子仍在框内。结果标准为 `released_inside_tray_and_return_home`。

旧抓取/搬运/放稳检查保留在结果的 `strict_quality_evaluation`，不与新任务标准混淆。之前 `xyshift_20261027_225137_retry04` 的58秒示范仍不含回位，原数据和验收记录保持不变。

## 当前布局启动命令

以下命令仅适用于本批导出的同一布局，输出目录必须不存在。调整任意物体后，重新读取场景并计算IK，不能直接复用旧坐标或只改哈希。

```bash
conda activate orcalab_9_1
cd ~/SY/OrcaPlayground
capture_root="$HOME/SY/datasets/tray_transfer/layout_20261027_232508"
capture_id=$(date +%Y%m%d_%H%M%S)
python -m examples.euler.tray_transfer.scene_pick \
  --plan "$capture_root/scene/plan.json" \
  --output "$capture_root/diagnostics_$capture_id" \
  --online --contact-diagnostics \
  --record-episode "$capture_root/episode_$capture_id" \
  --task-index 1
```

先打开对应布局并进入运行场景；脚本从布局初始状态执行，执行期间不要再次点击Play或调整物体。每步等待两路相同仿真索引的画面，再执行动作。记录为50Hz仿真时间，墙钟录制通常更慢。

## 保存内容

- `episode.npz`：15维实测state（右臂7+夹爪8）、9维绝对action（右臂7+a/d）、展开15维指令、锁存前请求、阶段、索引和时间。
- `frames/head`、`frames/right_wrist`：224×224 RGB PNG，按原宽高比缩放补黑边，无亮度增强。
- `metadata.json`：英文任务、状态/动作顺序、相机配置、判定与来源；`camera_frames.json`保存原生时间戳和索引。
- `--output` 指定的诊断目录：轨迹、执行源码、计划、模型信息与实际检查结果；与 episode 目录分别保存。

夹爪a/d为连续控制参数，展开为 `[0,0,a,d,0,0,-a,d]`，不是两个开关或牛顿单位的力。每帧为先观察后执行的 observation_t/action_t。

`--task-index 0/1/2`选择同一任务的三句英文表述，整条episode自动关联同一句，无需手动配对。

每个新布局先采1条。原始画面保持不变，亮度/对比度增强留到训练阶段；避免改变浅蓝色身份，不能代替位置、角度和干扰物布局的变化。后续扩充时按原布局划分训练/验证，再做增强；当前 8 条全部用于试训，尚无独立验证集。

原始 PNG+NPZ 和其中的 `training_ready=false` 保留不变；该标记描述原始记录，不代表转换没有完成。LeRobot 数据是单独生成的副本，不能用“格式通过”代替模型闭环成功率。返回 [代码框架](README.md)。
