# Demo 1 v1：G1 绿色桌子定点导航

这个分支只保留一个演示任务：在手动打开的 `demo1_v1.json` 场景中，代码生成
G1，让它沿点目标方向前进；头部 RGB 识别绿色桌面后向画面较空一侧绕行，最后
到达桌子另一侧的固定坐标。

## 当前场景参数

```text
Layout: /home/jason77/SY/demo1_v1.json
绿色桌子: industrial_workbench_1，约 (4.7, -5.0)
G1 出生点: (4.7, -8.0, 0.0)
出生朝向: yaw=90°，面向世界坐标 +Y
点目标: (4.7, 0.0)，位于绿色桌子正对面
到达半径: 0.35m
默认时间: 4000 个 50Hz 控制步，约 80 秒
相机: camera_head RGB，端口 7072
```

起点与目标的直线穿过绿色桌子，因此如果 RGB 避障没有生效，G1 会直接接近或
碰到桌子；正常运行时，绿色像素占比会触发减速、侧移和转向。

## 运行

1. 在 OrcaLab 中打开 `/home/jason77/SY/demo1_v1.json`。
2. 点击运行并选择 `No simulation program (manual launch)`，确认绿色桌子位于约 `(4.7, -5.0)`。
3. 在终端执行：

```bash
cd /home/jason77/SY/OrcaPlayground
conda activate orca

python examples/euler/g1_vision_nav/run_green_table_crossing.py
```

脚本保留手动打开的 Layout，只发布 actor `g1_navi`。导航算法仍读取
`camera_head:7072` RGB，但默认不打开浏览器；第一视角直接在 OrcaLab 中观察。
需要调试像素输入时，可用 `--camera-window` 打开浏览器预览。

常用参数：

```bash
# 临时启动浏览器预览
python examples/euler/g1_vision_nav/run_green_table_crossing.py --camera-window

# 启动浏览器并使用另一个本地预览端口
python examples/euler/g1_vision_nav/run_green_table_crossing.py \
  --camera-window --camera-window-port 8877

# 临时覆盖出生点、目标或运行时间
python examples/euler/g1_vision_nav/run_green_table_crossing.py \
  --spawn-x 4.7 --spawn-y -8.0 --spawn-yaw 90 \
  --goal-x 4.7 --goal-y -2.5 --num-steps 4000
```

最终 RGB 样本默认保存到 `/tmp/g1_green_table_goal_rgb.png`。

## 控制结构

```text
世界坐标目标 + G1 实时位姿
             ↓
   50 Hz 重算桌子正对面点目标 vx / vy / yaw
             ↓
camera_head RGB → 中央绿色达到 10% 时保持 0.08m/s 小步旋转（yaw=±0.50rad/s）
             ↓ 中央恢复清晰
        立即重新使用当前位置到目标点的 vx/yaw

普通桌面接触不会永久锁死控制指令；确认摔倒仍会触发紧急停止。
             ↓
      冻结 G1 locomotion ONNX
```

绿色检测只是当前固定 Layout 的可解释基线，不代表通用障碍检测，也不能从单目
RGB 得到真实米制距离。

## 核心代码导读

```text
run_green_table_crossing.py   唯一演示入口；固定起点、目标并组装整条链路
layout_scene.py               通过 AddActor 发布 g1_navi，注册 Studio 相机组件
g1_camera_stream_env.py       激活 camera_head 并读取 7072 RGB
point_goal_navigator.py       根据实时位置和朝向重算点目标速度指令
green_table_navigator.py      检测绿色桌面并临时修正点目标指令
g1_vision_nav_env.py          连接上层导航与冻结的 G1 ONNX 运控模型
g1_green_table_env.py         汇总本演示的在线验证指标和第一视角信息
```

入口名称使用“green table crossing”，因为本分支并不是通用视觉避障系统，而是
一个条件明确、可重复录制的绿色桌子穿越演示。

本次只调整职责命名，没有改变已经在线验证过的控制参数：

```text
run_visual_avoidance.py      -> run_green_table_crossing.py
visual_avoidance.py          -> green_table_navigator.py
g1_visual_avoidance_env.py   -> g1_green_table_env.py
g1_camera_validation_env.py  -> g1_camera_stream_env.py
```
