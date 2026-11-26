# 闭环评测与结果检查

完成环境安装、资源同步、布局加载和双相机检查后，在示例目录的 `orca` 环境运行：

```bash
python reproduce.py evaluate \
  --checkpoint models/pi05-025000 --tokenizer models/tokenizer \
  --output outputs/eval-pad-10 --trials 10 --image-mode pad
python reproduce.py audit \
  --evaluation outputs/eval-pad-10 --output outputs/eval-pad-10-audit.json
```

每轮使用新的输出目录。运行时保持 Studio 和仿真开启，
同一场景只运行一个采集或评测程序。

## 控制与成功条件

模型读取头部、右腕图像和 9 维状态，每次预测 50 个绝对动作，
以 50 Hz 执行，之后再次观察并预测。统一使用命令中的 `pad`，
将两路图像保持比例缩放至 224×224 并补黑边，与训练转换一致。
脚本自动处理并记录图像模式，无需额外运行预处理或电脑录屏软件。

原始动作先限幅至场景执行器范围，再执行；两侧夹爪分别使用各自的模型输出。
碰撞、关节跳变、左臂移动等保护仍会中止试验。
没有专家轨迹或预设搬运动作接管模型控制。
左臂保持 `[-0.3, 0.3, 0, 1.5, 0, 0, 0]`，腰部 pitch 为 `0.35 rad`。

一次成功需要同时满足：

- 物品抬起超过 8 cm。
- 整个盒子进入木框并接触框底。
- 模型主动张爪，物品脱离夹爪。
- 物品速度低于 0.02 m/s，连续稳定 50 个控制周期，即 1 秒。

只夹起来、一直握着、搭在框边或碰框翻落均不算成功。
默认最多执行 3000 个控制周期；超时、保护中止和推理错误保留为失败。
成功率使用全部请求试验次数作为分母。

## 输出位置

| 文件 | 内容 |
| --- | --- |
| `summary.json` | 请求数、完成状态和逐次结果 |
| `settings.json` | 种子、图像模式、执行器限幅范围和保护参数 |
| `provenance.json` | 代码、模型、tokenizer 和环境来源 |
| `trial_*/cameras.mp4` | 头部与右腕双相机录像 |
| `trial_*/initial_*.png` | 每次初始相机图像 |
| `trial_*/predicted_chunks.npz` | 原始模型预测块 |
| `trial_*/trajectory.npz` | 状态、原始动作 `action_raw`、执行命令 `action` 和物理测量 |
| `trial_*/result.json` | 成功标记、结束原因及经过限幅处理的动作行数 |

`policy_actions_replaced` 记录限幅改变的动作行数，`expert_takeover` 为 `false`。
审计分别核对原始动作与预测块、执行命令与声明的限幅公式，
并检查视频帧数、完整试验分母和稳定放置条件。
旧版未经过动作处理的记录仍可核验。

查看全部录像，结合动作记录确认抓取、搬运、主动松爪和稳定放置。
审计通过表示记录一致，不替代人工录像检查。
当前场景的物品与木框位置固定，结果仅适用于实际测试条件。
