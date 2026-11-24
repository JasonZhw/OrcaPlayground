# 场景、资源与自动采集

## 可视资源

`assets/scene_v9.json` 是原V9布局，包含以下资源引用：

| 名称 | tech_name | 用途 |
| --- | --- | --- |
| SimpleMovement_Screw | simplemovement_screw | 基础关卡 |
| unitree_g1_gripper | g1_pick_with_gripper | 带夹爪G1 |
| Kitchenware | kitchenware | 木框 |
| Warehouse_EmbodiedCompetition_2026Hangzhou | warehouse_embodiedcompetition_2026hangzhou | 桌子与蓝色Intel盒子 |

安装OrcaLab26.9.1后，在有GPU显示的桌面启动 `python reproduce.py studio`。
登录自己的OrcaLab账号，在资源管理界面订阅/同步以上包，
再加载布局。布局中保留了原逻辑资源标识和相机挂载。
访问受资源服务权限与可用性约束；本项目没有内置平台账号或授权凭据。
现有布局未记录可视包的不可变内容哈希，因此资源更新可能影响画面。

全部资源就绪后，`python reproduce.py studio --load` 调用公共
`load_layout(layout_path=...)` 和 `start_simulation(program_name="external")`。
这些参数已与26.9.1的CLI帮助及公共方法签名核对。
该命令会切换当前OrcaLab场景，应在空闲实例运行。

物理XML包含碰撞体、质量、摩擦和夹爪网格，但不含完整纹理资源。
离线MuJoCo几何渲染不能替代训练时的OrcaStudio相机图像。
没有同步上述视觉资源时，应先完成资源加载，再采集或评测。

## 物理与控制

`prepare-scene` 根据完整接触场景固定机器人腰腿自由度；
保留双臂、夹爪、相机，腰pitch为0.35。
左臂参考姿态为 `[-0.3,0.3,0,1.5,0,0,0]`，右臂由策略控制。
生成器验证物体、刚体、碰撞体、相机、站点的位置及碰撞参数，
并将运行时绝对资源路径写入已忽略的 `runtime/models/`。

控制器继承 `OrcaGymEulerEnv`，通过公共API执行仿真。
每个50Hz动作由20个1ms物理子步执行，
保留原关节位置控制、重力补偿与夹爪连杆控制。
环境地址从 `ORCAGYM_ADDR` 读取，默认 `localhost:50051`。
相机为 `head_cam` 与 `camera_right`，640×480 RGB，
使用7070/7072本地流端口及匹配的仿真索引同步。

## cuRobo采集

先运行 `prepare-scene` 和 `studio --load`，随后：

```bash
python reproduce.py collect --output data/new_v9 --count 200
```

每条示范从同一场景复位开始。完整阶段：
张爪、预抓取、接近、中心稳定、闭爪、双侧接触保持、抬起、搬运、
框上对准、下降、松爪、撤离和稳定。
规划器验证标定FK误差小于0.5mm；采集过程校验抓取对准、
双侧接触、抬升高度及搬运滑移。录像在每个动作之前采样。
每条输出 `cameras.mp4`、`trajectory.npz` 和 `result.json`。

原V9采集接受条件为物体中心落在框区域、低速、抬起足够并脱离夹爪；
它弱于闭环评测的“整个碰撞盒在框内+框底接触+持续稳定”判定。
采集成功不能直接计为模型推理成功。

便携版本使用原采集器实际导入的 `planning_world_collection.yml` 障碍物快照，
并保留 `planning_world_v9.yml` 作为诊断参考。
后者单独测试会在起始状态触发碰撞，不能直接替代原采集器的规划配置。
物理碰撞仍由实际V9场景处理，规划障碍物快照不等于场景全部几何。
本版去掉旧训练任务的状态依赖与并发显存限制。
放置偏置保留V9的框坐标 `(0.11,-0.04)`，不是后续居中纠正实验。
cuRobo的随机搜索、环境版本和硬件可能改变轨迹长度；
重新采集不保证得到原200条相同的数据。

已有轨迹会重新验证状态/动作与录像对齐，只统计成功条目。
连续两次失败会停止，检查日志和录像后可以原目录继续。
采集与推理使用同一个运行目录锁，不能同时占用场景和相机。
`PI05_V9_WORK` 可设置运行目录；它只影响运行产物，不更改源资产。
