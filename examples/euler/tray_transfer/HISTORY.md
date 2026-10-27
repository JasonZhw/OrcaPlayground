# 历史开发记录：蓝件验证到包装盒采集

本文件保留 2026-10-28 整理前的 README 原文，供追溯实验参数和失败原因。下文的“当前”“下一阶段”和“尚未转换”等表述属于当时记录，不能作为现状或最新启动指令。

当前任务、代码结构和入口见 [README](README.md)，实际采集见 [CAPTURE](CAPTURE.md)，已完成的数据转换见 [LEROBOT](LEROBOT.md)。

---

# G1：物体抓取与木托盘搬运

目标是在同一回合中把待操作物体释放到左侧木托盘内。搬下一件时不重置场景，已经放下的物体要继续作为障碍物参与规划。

**当前进度：2026-10-27，新位置已采集1条入框并复位示范，共3395组同步样本、67.90秒。** 本批位于 `~/SY/datasets/tray_transfer/layout_20261027_232508/`，使用新布局IK和5°倾斜抓姿。当前任务判定是物体入框、松爪、右臂回位，不再要求盒子静止或仅由托盘底面支撑；严格质量诊断单独保留。此前7条记录保留，其中上批XY示范不含回位。数据尚未转换LeRobot，详见[采集说明](CAPTURE.md)。

**2026-10-27 已在真实 OrcaLab 窗口完成100克 Intel 包装盒的抓取、横移、下放、松爪和退手，单件搬运判定通过。** 本次在线记录为 `outputs/new_layout_20261027_184921/live_transfer_ik_fix2/`，全程73秒；松爪前持续夹持，最终盒子由托盘底板承重、与夹爪分离。途中仍有约30.89 mm相对滑移、16.48°相对转动，最大指尖压入11.55 mm，保留为诊断；`training_data: false`，尚未用于训练。详细坐标、失败原因和结果见 [包装盒抓取与搬运记录](NEW_LAYOUT_PICK_REPORT.md)。下文蓝件命令用于复现旧场景，不能直接套用到新布局。

旧蓝件布局曾实现场景读取、候选位置几何检查，以及固定底座的离线物理测试。**2026-10-27早期记录：官方 joint2 控制可以把蓝件抓离桌面，但该旧场景的完整搬运未通过。** 只固定 pelvis 时腰部明显前倾；额外固定腰部后，蓝件仍在继续抬高时滑脱。两轮均未放入托盘，没有生成训练数据。旧场景结果、回放位置和复查命令见 [后续搬运验证](BLUE_TRANSFER_REPORT.md)；此前夹爪控制判断的纠正见 [蓝件试抓记录](BLUE_PICK_REPORT.md)。

## 当前开发：搬进托盘

最新计划位于本机 `~/SY/datasets/tray_transfer/layout_20261027_232508/scene/plan.json`，按当前场景重新求解5°倾斜抓取，已执行至回位。具体任务标准与采集结果见CAPTURE.md。下述包含此前完整流程与历史诊断，动作顺序为：

**左臂放下 → 抓取 → 对齐并渐进加紧到d=1.45 → 抬起 → 按实际持物姿态规划 → 抬过栏杆并连续转腕 → 横移进框 → 下放 → 松爪 → 抬手退出 → 确认盒子在框内 → 右臂回默认位置。**

回位阶段 `return_home → home_hold` 按实际场景检查空手回程，只恢复布局启动时右臂的7个关节目标，夹爪继续张开，左臂保持放下；不会重置场景或搬回托盘里的盒子。最新记录已执行到回位，最大关节误差约0.0000212 rad，回位后盒子仍在框内。全流程82.9秒，去掉15秒准备阶段后录制67.9秒。此前批次的时长和判据见各自数据说明。

下一阶段的两相机、任务指令和随机化方案见 [三盒任务与采集设计](TASK_DESIGN.md)。

搬运期间保持八路夹爪目标 `[0,0,0.28,1.35,0,0,-0.28,1.35]`，只有明确进入 `release` 阶段才解除保持。开爪前重新读取实际物体边界，要求物体整体在托盘内且底面接近底板；如果偏离，保持闭合并报告位置问题。抓取后的两秒是阶段过渡，不是重新进行20秒保持实验。

规划器使用当前场景的独立物理模型做静态FK/IK和碰撞采样，实际仿真中的物体保持原物理属性。持物姿态来自抓起后的真实状态；先保持实测姿态抬高，再连续转至绕世界Y轴−5°的搬运候选姿态，修复原搬运路径IK无解（下放段不可达）的问题。参考文件中的坐标不直接替代运行时测量。栏杆上方留35 mm名义间隙、托盘内边留25 mm、名义释放高度距底板5 mm。这些是路径参数，不是新的指尖压入限制。当前仍为IK路点控制，尚未接入cuRobo。

### 在 OrcaLab 中启动搬运开发版

打开当前包装盒布局，进入运行状态，使gRPC `127.0.0.1:50051` 可用。以下命令从布局初始状态执行整个流程：

```bash
conda activate orcalab_9_1
cd ~/SY/OrcaPlayground
tray_run="$HOME/SY/datasets/tray_transfer/layout_20261027_232508"
python -m examples.euler.tray_transfer.scene_pick \
  --plan "$tray_run/scene/plan.json" \
  --output "$tray_run/transfer_preview_$(date +%Y%m%d_%H%M%S)" \
  --contact-diagnostics \
  --online --preview --development-preview
```

初次闭合目标自动从计划读取为0.97，最终加紧1.35写在 `tighten` 阶段中；不需要再传 `--close-drive`。不要把最终加紧值1.35用于初次闭合，本次旧启动参数曾因此触发指根碰撞。`--development-preview`按本次开发要求跳过“必须先有完整离线动力学结果”的条件；仍核对场景和资产、在实际抓起后求解路径并检查运行时碰撞。若只想运行离线副本，去掉最后一行的三个在线参数即可。

运行中不再按Play。`Ctrl+C`可停止；结束后画面保留最后姿态，不自动恢复此前布局。抓后计算路径时物理不推进，画面短暂停留属预期。输出包含实际生成的 `transport_plan.json`、逐步轨迹和分开统计的抓取/搬运/放置结果，始终 `training_data: false`。

此前在线执行的抓取、途中夹持和最终放置均通过：途中两指最小法向力1.72 / 1.77 N；最终物体距托盘内边至少44.68 mm，底板支撑力约0.981 N、夹爪接触力为0。真实窗口截图见 [最终放置](outputs/new_layout_20261027_184921/live_transfer_ik_fix2/orcalab_final.png)。此前完成3个布局下的6条完整流程采集；另有当前XY布局1条任务示范，录到放稳，不含回位。倾斜抓取批次的翻转诊断仍保留，目标物角度变化尚未采集。

## 复现上一版单件抓取诊断

新布局的离线复核入口为 `scene_pick.py`。以下命令复跑加深进近、指面对齐、渐进加力及20秒悬空保持；`--contact-diagnostics` 采用连续夹持判定，指尖压入仅作记录。这是抓取验证入口：

```bash
conda activate orcalab_9_1
cd ~/SY/OrcaPlayground
python -m examples.euler.tray_transfer.scene_pick \
  --plan examples/euler/tray_transfer/outputs/new_layout_20261027_184921/pick_retention_ready/plan.json \
  --output examples/euler/tray_transfer/outputs/retention_recheck \
  --close-drive 0.97 --contact-diagnostics
```

输出目录必须尚不存在。默认离线，不控制当前 OrcaLab；在线测试需要匹配场景、路径、控制参数、判定模式与源码的离线记录。该计划先以0.97闭合，再在1.5秒内对齐到 `[0,0,0.28,0.90,0,0,-0.28,0.90]`，保持2.5秒后，用3秒将两侧joint2目标从0.90增至1.20，随后锁存全部八路指令直至结束。这里是伺服目标，不是直接赋值的关节位置或牛顿单位的夹持力。同时记录接触坐标、实际关节、执行器力和伺服输入。

旧单抓记录 `pick_retention_ready` 按当时夹持标准通过。当前源码已扩展搬运，其哈希与旧记录不同；不能直接把旧记录作为当前实现的在线依据。单抓如需在线复现，先用上面的命令生成当前代码的离线记录，再通过 `--offline-evidence` 指定该新目录。当前搬运开发使用前一节的直接预览命令。

### 旧蓝件布局

在 `feature/Synthetic_Data` 分支、`orcalab_9_1` 环境中执行：

```bash
conda activate orcalab_9_1
cd ~/SY/OrcaPlayground
python -m examples.euler.tray_transfer.blue_pick \
  --model-xml examples/euler/tray_transfer/outputs/gripper_calibration_20261027/current_raw.xml \
  --profile original
```

这条命令**只运行离线 Euler 物理副本，不会让 OrcaLab 窗口中的机器人运动**。`original` 是默认配置，保留资产原有碰撞、接触和质量参数；测试中固定 pelvis，并通过关节伺服运动。当前复测已抓起蓝件，但因压入超过 2 mm 返回退出码 2；这表示严格质量检查未通过，不是夹爪不能工作。

右夹爪只对两侧 `joint2` 执行器发送共同输入，张开为 `-1`、参考闭合为 `2`，其余六个执行器输入保持 `0`。输入不是可以直接赋给 qpos 的关节位姿。映射见 [gripper_control.py](gripper_control.py)，依据是[官方控制参考](https://github.com/openverse-orca/SouthGrid/blob/7e98424bc3726b82f73aad66dc1e714e8dbe60fb/src/conf/g1_pick_osc_conf.py#L90-L97)。

每次生成独立的 `outputs/blue_<配置>_<时间>/`，含模型改动清单、资源哈希、逐控制步接触与位姿、名义动作及末子步伺服输入、成功判定。结果是物理诊断，不能直接作为 LeRobot 数据集。

以上 XML 和 `assets/` 是本机已有的场景导出，未上传 Git；新的仓库克隆不会自带它们。路径文件绑定当前导出 XML、STL 内容、初始物体位姿和机器人关节；换物体、改布局或重新导出后需要重新标定并生成路径，不能只改校验哈希后继续跑。

## 1. 检查当前场景

在 OrcaLab 26.9.1 中打开搭好的场景，然后在仓库根目录执行：

```bash
conda activate orcalab_9_1
cd ~/SY/OrcaPlayground
git branch --show-current
python -m examples.euler.tray_transfer.inspect_scene
```

开发分支应为 `feature/Synthetic_Data`。工具读取 `127.0.0.1:50051`，只调用公开查询 RPC，不创建环境，不启动、暂停、重置或推进仿真，也不修改物体和机器人。后续控制环境使用 `OrcaGymEulerEnv`；这里单独查询是为了避免环境构造时的初始化影响当前场景。

每次运行在本目录 `outputs/<时间>/` 保存：

- `scene_snapshot.json`：场景模型、碰撞体、质量、实际 body 姿态、夹爪约束和相机名称。
- `report.json`：物体顺序、放置候选、坐标变换结果、几何检查和仍需验证的项目。

这些运行产物已被 Git 忽略。查询是顺序执行的，不是同一个仿真时刻的原子快照；请在场景静止时使用。读取相机名称不代表已经取得图像，界面显示 Running 也不证明物理时间正在推进。

`Geometry preflight: PASS` 仅表示**两个物体的候选碰撞包围盒位于配置的托盘内部、互不重叠，并且未与托盘碰撞体包围盒重叠**。它不证明夹爪可达、路径无碰撞、能承重或放置稳定。检查失败返回退出码 2。

可离线复查已有快照：

```bash
python -m examples.euler.tray_transfer.inspect_scene \
  --snapshot examples/euler/tray_transfer/outputs/<时间>/scene_snapshot.json
```

## 2. 配置放置区域

编辑 [task.json](task.json)：

- `objects` 仍保留早期双物体预检的灰件→蓝件顺序；当前 `blue_pick` 单独测试蓝件，不使用该顺序。灰件替换后再更新连续任务配置。
- `body_suffix` 匹配物理模型中的 body 名称；若同名资产出现两次，工具会报歧义，需要填完整名称。
- `placement_center_xy` 是**托盘 body 坐标系中，物体碰撞包围盒中心的 XY**，单位米。不是世界坐标，也不是夹爪 TCP。
- 两个初始候选中心为灰件 `[0.19, -0.10]`、蓝件 `[0.09, -0.075]`，靠近当前机器人一侧；这些只是几何候选，须通过手臂可达性检查后才可用于动作规划。
- `inner_xy_min/max` 是当前托盘从栏杆和立柱内侧测得的保守内腔，`edge_clearance_m` 额外留出 2 cm。换托盘或改变尺寸后需要重新测量。
- `floor_size_xyz_m` 是底板完整尺寸。碰撞体 UUID 会随加载改变，因此按底板的尺寸与方向匹配，不保存易失的 UUID。
- `release_clearance_m` 让物体候选底面高于底板 5 mm，实际接触和稳定性仍需物理测试。

工具根据当前物体姿态生成候选，保持物体的世界朝向，并计算正确的 body 原点；同时输出世界坐标与 G1 pelvis 基座坐标。两者不能和编辑器中的机器人 actor 原点混用。碰撞形状来自仿真模型，不使用截图或视觉包围盒推算。

当前目标物和托盘都是 primitive 碰撞体，支持 box、sphere、cylinder、capsule、ellipsoid。碰撞 mesh 或带子 body 碰撞体的新目标会明确报错，需要对应的几何适配，不能把 `geom_size` 当作 mesh 外形。

## 3. 后续接入专家控制

### 已验证的左臂放下姿态

2026-10-27 已在当前布局的离线副本及 OrcaLab 在线显示中执行：**先收肘 → 把手撤到桌沿外 → 放到身体左侧 → 保持 3 秒**。直接向下转肩会碰托盘栏杆，不能跳过前面的撤手过程。动作参数保存在 [left_arm_down_pose.json](left_arm_down_pose.json)。

控制使用 `BluePickEnv.step()`／Euler 的 `do_simulation()` 发送关节目标，并通过 `render()` 同步到 OrcaLab。只固定 pelvis，腰部维持原有伺服控制；15 秒动态轨迹的 50 Hz 样本中，未发现左臂与外部的接触或控制饱和。结束后通过只读 RPC 核对左手位置，与执行结果一致，OrcaLab 保持暂停。原有夹爪内部接触单独记录，不属于本次外部避碰验收。

这是运行时待机姿态，未写入机器人资产或 Layout 的默认关节配置。重新加载／重置后，需要在抓取前重新执行；新布局必须重新检查路径。后续右臂控制仍应持续给左臂发送已验证的保持目标，不能每步把整组动作清零。此次没有验证右臂持物经过放下后的左臂时是否无碰撞。

本机证据保存在 `outputs/left_arm_down_20261027/`（离线回放）与 `outputs/left_arm_down_live_20261027/`（在线结果、执行脚本、只读回查）。这些诊断文件被 Git 忽略。

### 后续任务

1. 在当前单件在线成功的基础上，减少搬运中约30.89 mm滑移及16.48°转动，再复核抓点与搬运姿态。压入继续记录；一次固定布局成功不代表随机化成功。
2. 根据实际落点优化进框和释放位置，并测试其他物体；旧蓝件失败与本次包装盒成功分别保留。`--fix-waist`是旧蓝件离线夹具选项，本场景只固定pelvis。
3. 将当前机器人关节、TCP、夹爪及障碍物映射到 cuRobo，先验证单件抓取→抬升→搬运→释放。不能直接将 stock G1 配置当作本场景四连杆夹爪模型。
4. 第二件规划时更新已放好的第一件位置；最终检查两件都在托盘内、脱离夹爪并稳定一段时间。两件直立高度均超过托盘内深，不要求最高点低于栏沿。
5. 基础动作成功后再添加每回合起始位置随机化、重规划及数据录制。专家可用仿真真值，π0.5 输入保持为图像、机器人状态和任务文本。

当前保留此前3个布局的6条完整流程记录，以及新XY布局1条录到放稳的用户验收记录，尚未转换为LeRobot；每批目录分别保存任务判定和存储检查结果。不要将候选 body 位置直接发送给夹爪，也不要把几何检查通过的结果记成成功 episode。

## 当前夹爪的离线核对结果

2026-10-27 从当前场景导出模型，并按稳定 body/joint 名称映射现场 qpos。离线 `mj_forward` 后，四个指端 body 的位置与现场查询一致。使用编译后的 mesh 核对，当前闭合右指面间的最小距离约 0.059 mm；连杆原点相距约 44.62 mm，不能把它当作夹持净空。

右夹爪闭合指面最近点的中点，在 `ee_center_site_r` 坐标系中约为：

```text
[+0.073295, -0.000056, -0.003074] m
```

因此该 site 不能直接当作指面接触中心。这个测量只对应当前闭合姿态，开合时还需要重新计算接触中心和方向，不能把它永久硬编码成通用 TCP 偏移。

早期离线几何标定以两侧 joint1 共同角度为参数，求解其余六个角并保持指面平行：`u=0.4/0.5/0.65 rad` 对应约 `54.02/62.46/72.02 mm` 的预张开净空。**这些是静态几何候选，不是当前执行器指令。** 运行控制采用上述 joint2 映射，并用实际运动后的指面姿态验证抓取；此前八个关节同时追踪标定角的方案失败，不能推广为原始夹爪失效。

旧蓝件试抓使用蓝件窄边；灰件未测试。当前任务已更换为Intel包装盒，换物体后仍需重新测量形状、质量和抓取位置。

## 验证

```bash
python -m unittest discover -s tests -p 'test_tray_transfer*.py'
python -m unittest discover -s tests -p 'test_blue*.py'
python -m unittest tests.test_grasp_alignment tests.test_scene_pick_evidence tests.test_scene_transfer_result
python -m ruff check examples/euler/tray_transfer tests/test_tray_transfer*.py tests/test_blue*.py tests/test_grasp_alignment.py tests/test_scene_pick_evidence.py tests/test_scene_transfer_result.py
```

测试覆盖碰撞体尺寸、坐标变换、放置区域、夹爪控制映射、伺服补偿、显式腰部夹具、异常结果记录，以及抓起／搬运／放置判定；不依赖 GPU 或运行中的 OrcaLab。单元测试通过不替代实际物理轨迹的验收。
