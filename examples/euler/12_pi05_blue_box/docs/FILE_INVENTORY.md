# 文件清单

| 路径 | 用途 |
| --- | --- |
| `reproduce.py` | 统一阶段入口：下载、场景、采集、转换、训练、推理、评测 |
| `environment.yml`、`requirements*.txt`、`scripts/bootstrap.sh` | orca 环境与依赖安装 |
| `configs/sources.json` | 数据、模型、tokenizer 固定 revision，模型大小及哈希 |
| `configs/visual_resources.json` | 四个视觉包的名称与内容版本固定状态；未知哈希保留为空 |
| `configs/train_config.saved.json` | 原始完整训练配置，保持原文件字节 |
| `configs/input_files.sha256.json` | 七份原始训练材料的 SHA256 |
| `assets/scene_v9.json` | OrcaStudio V9 布局及包引用 |
| `assets/scene_contact_v9.xml` | 完整 MuJoCo 接触场景 |
| `assets/scene_single_target_v9.xml` | 保留的固定底座场景参考；运行时重新构建 |
| `assets/*.stl` | 原场景夹爪的九个网格 |
| `assets/calibration.json`、`g1_scene_right_arm.*`、`planning_world*.yml` | 坐标标定、机器人运动学、原采集规划障碍物与诊断参考 |
| `simulation/build_model.py`、`environment.py` | 固定腰腿的模型生成与 Euler 公共 API 控制器 |
| `simulation/planner_v9.py` | 固定版本 cuRobo 新 API 的运动规划 |
| `simulation/collect.py`、`collector_base.py`、`collect_200.py` | 专家动作、接触门控、200条采集与继续采集 |
| `simulation/camera_capture.py` | 头部/右腕同步取帧与退出时释放服务端推流 |
| `simulation/placement.py`、`evaluate.py` | 整体入框判定与策略闭环评测 |
| `scripts/download_assets.py` | 固定版本下载与权重哈希校验 |
| `scripts/convert_raw.py`、`image_preprocessing.py` | 原始录像/NPZ 转 LeRobot v3.0 |
| `scripts/prepare_train.py`、`train.py` | 数据预检、保存配方、调用官方 LeRobot 训练器 |
| `scripts/infer.py`、`policy_runtime.py` | 单观测动作预测与常驻模型运行时 |
| `scripts/start_studio.py`、`prepare_scene.py` | 启动/加载可视场景，生成运行时路径 |
| `scripts/check_environment.py`、`collect_versions.py` | 运行环境、CUDA、资源与版本清单 |
| `simulation/check_scene.py` | 初始场景复位、同步双相机 PNG、帧有效性和人工审核清单 |
| `scripts/evaluation_metadata.py`、`audit_evaluation.py` | 评测来源记录、权重/tokenizer/代码哈希与结果证据审计 |
| `scripts/verify_public.py`、`smoke_physics.py` | 隐私/哈希/便携性检查与离线物理检查 |
| `reference/training/` | 原转换程序、训练入口、包清单、info.json、stats.json |
| `third_party/openpi/` | 官方图像函数原文件及许可证 |
| `third_party/curobo/` | 上游许可证；库本体按固定提交安装 |
| `tests/` | 配方、预处理、放置判据、限幅记录核验、依赖与相机生命周期测试 |
| `docs/ACCEPTANCE.md` | 安装、闭环评测、证据发布及重新训练的分阶段验收流程 |

不收录原机器连接脚本、个人运行日志、云盘同步清理程序。
Hugging Face 数据和权重保留在原仓库，由下载脚本获取。
OrcaStudio 的纹理、可视模型和引擎包通过资源系统同步，不包含在这些物理 XML 中。
