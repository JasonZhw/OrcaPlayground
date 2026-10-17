# 药品推理动作段对照

本轮只测试现有两万步π0.5模型每次连续执行1、2、4步的差别。每组四种药品各10次，共120次；5Hz、150步上限、场景、图像与验收条件保持一致。无专家动作回退，不补采、不重训，也不改160条合格数据和旧模型。

三组使用相同的40个初始摆放seed。它们是此前测试过的摆放，现在用于配对调参，不是独立新测试。每组重启策略服务，每种物品重新加载同一场景；策略随机采样仍存在波动，单轮小样本差异不能证明显著提升。选定配置后还需要新seed独立验收。

## 查看进度

```bash
systemctl --user status sy-medication-chunk-compare.service --no-pager
journalctl --user -u sy-medication-chunk-compare.service -n 10 --no-pager
```

输出根目录：`/home/pc/ZXJ/SY/data/medication_sorting/policy_evaluations/chunk-comparison-20261016-v2`。

无`-v2`后缀的目录是已停止的预启动记录；它不计入本轮120次对照，也不会覆盖。

- `configuration.json`：固定参数、模型/场景/代码指纹和配对种子。
- `status.json`：当前组别、物品和正在生成的summary路径。
- `progress.json`：已结束并复核的批次计数。
- `chunk_1/`、`chunk_2/`、`chunk_4/`：逐物品summary、逐回合报告、RGB和真实动作轨迹。
- `pipeline_logs/`：策略加载、场景加载、评估与备份日志。
- `comparison.json`：仅全部120回合完成且模型证据一致时生成，包含各组完整成功率、抓起数、未抓起数和抓起后失败数。

完整成功数最高的配置作为后续独立验收候选；同分时选较大动作段，减少推理调用。不能只挑成功回合，也不能把抓起成功率当成分类入筐成功率。

新输出目录必须不存在，已有目录不会覆盖。任务不自动重放中断回合；异常保留现场后需要先检查，不能直接重跑原目录或启动旧采集训练入口。仅管理本任务创建的进程，不接管外部窗口或模型服务。
