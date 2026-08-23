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
2. 点击运行，确认绿色桌子位于约 `(4.7, -5.0)`。
3. 在终端执行：

```bash
cd /home/jason77/SY/OrcaPlayground
conda activate orca

python examples/euler/g1_vision_nav/run_visual_avoidance.py
```

脚本保留手动打开的 Layout，只发布 actor `g1_green_table_demo`。默认会在浏览器
打开 `http://127.0.0.1:8765`，显示实时 RGB、目标距离、指令和绿色区域占比。
关闭或刷新网页不会停止导航；若浏览器没有自动打开，可以手动访问终端打印的地址。

常用参数：

```bash
# 不启动浏览器预览
python examples/euler/g1_vision_nav/run_visual_avoidance.py --no-camera-window

# 使用另一个本地预览端口
python examples/euler/g1_vision_nav/run_visual_avoidance.py --camera-window-port 8877

# 临时覆盖出生点、目标或运行时间
python examples/euler/g1_vision_nav/run_visual_avoidance.py \
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
