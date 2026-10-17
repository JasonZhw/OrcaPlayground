"""四药品推理的离线契约测试；替身不连接服务、不推进真实仿真。"""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from tools.medication_sorting import infer_policy, randomization, scaled_layout, task_specs


ROOT = Path(__file__).resolve().parents[1]
RANDOMIZATION_MODULE = 'tools.medication_sorting.randomization'


@pytest.mark.parametrize('help_requested', [False, True])
def test_import_and_help_do_not_load_runtime_dependencies(help_requested):
    script = '''
import builtins
import runpy
import sys
original = builtins.__import__
blocked = {'mujoco', 'scipy', 'orca_gym', 'grpc', 'openpi_client', 'jax', 'torch',
           'tools.medication_sorting.robot_control', 'tools.medication_sorting.run_expert',
           'tools.medication_sorting.rendered_rgb', 'tools.medication_sorting.collect_episodes'}
def guarded(name, *args, **kwargs):
    if name in blocked or name.split('.')[0] in blocked:
        raise AssertionError('不应加载运行依赖: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
if sys.argv[1] == 'help':
    sys.argv = ['infer_policy', '--help']
    runpy.run_module('tools.medication_sorting.infer_policy', run_name='__main__')
else:
    import tools.medication_sorting.infer_policy
'''
    result = subprocess.run([sys.executable, '-c', script, 'help' if help_requested else 'import'],
                            cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    if help_requested:
        for option in ('--target', '--profile', '--layout', '--randomization-profile'):
            assert option in result.stdout


@pytest.mark.parametrize('seed', [0, 413, 20261014])
def test_legacy_default_remains_exact(seed):
    offset, yaw = infer_policy.reset_randomization(seed)
    rng = np.random.default_rng(seed)
    np.testing.assert_array_equal(offset, rng.uniform([-.006, -.008], [.006, .008]))
    assert yaw == rng.uniform(-.025, .025)


@pytest.mark.parametrize('profile', ['legacy', 'tabletop_v1'])
def test_reset_delegates_seed_and_profile_to_shared_sampler(monkeypatch, profile):
    calls = []
    expected = np.array([.018, -.019])
    module = ModuleType(RANDOMIZATION_MODULE)

    def sample_reset(seed, profile='legacy'):
        calls.append((seed, profile))
        return expected.copy(), .24

    module.sample_reset = sample_reset
    monkeypatch.setitem(sys.modules, RANDOMIZATION_MODULE, module)
    offset, yaw = infer_policy.reset_randomization(831, profile=profile)
    assert calls == [(831, profile)]
    np.testing.assert_array_equal(offset, expected)
    assert yaw == .24


def test_tabletop_profile_never_silently_falls_back_to_legacy(monkeypatch):
    monkeypatch.setitem(sys.modules, RANDOMIZATION_MODULE, None)
    with pytest.raises(ModuleNotFoundError):
        infer_policy.reset_randomization(413, profile='tabletop_v1')


def test_shared_sampler_internal_import_failure_is_not_hidden(monkeypatch):
    module = ModuleType(RANDOMIZATION_MODULE)

    def sample_reset(seed, profile='legacy'):
        raise ModuleNotFoundError('共享模块内部依赖缺失', name='missing_sampler_dependency')

    module.sample_reset = sample_reset
    monkeypatch.setitem(sys.modules, RANDOMIZATION_MODULE, module)
    with pytest.raises(ModuleNotFoundError, match='共享模块内部'):
        infer_policy.reset_randomization(413)


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    """只替换外部边界，真实执行 main、限幅、接触计数及成功判定。"""
    harness = SimpleNamespace(
        output=tmp_path/'inference', target=task_specs.DEFAULT_TARGET,
        observations=[], commands=[], reset_calls=[], sample_calls=[], live_profiles=[],
        metadata_calls=0, robot_created=False, disconnected=False, camera_closed=False,
        lift=.07, grasp_frames=3, one_finger=False, unstable_at=None,
        live_error=None, reset_error=None, infer_error=None, state_dim=8,
    )
    harness.layout = tmp_path/'layout.json'
    harness.layout.write_bytes(scaled_layout.CAPTURE_LAYOUT.read_bytes())
    harness.metadata = {
        'robot_type':'unitree_g1_fixed_base_right_arm', 'state_dim':8, 'action_dim':8,
        'joint_units':'rad', 'joint_action_semantics':'absolute_position',
        'gripper_semantics':{'open':0., 'closed':1.},
        'observation_config':infer_policy.CAPTURE_CONFIG,
        'checkpoint_provenance':{'path':'/actual/checkpoints/1499', 'step':1499,
                                 'params_manifest_sha256':'a'*64},
    }
    harness.predictions = np.tile(np.r_[np.full(7, .12), .6], (4, 1))
    suffixes = list(dict.fromkeys(spec['body_suffix'] for spec in task_specs.TARGETS.values()))
    suffixes += ['_Box_Blue', '_Box_Purple', '_gripper_r_inner_link4', '_gripper_r_outer_link4']
    names = ['scene'+suffix for suffix in suffixes]
    bodies = [SimpleNamespace(id=i, name=name) for i, name in enumerate(names)]
    model = SimpleNamespace(
        nbody=len(names), body=lambda value: bodies[names.index(value)] if isinstance(value, str) else bodies[value],
        body_jntadr=np.arange(len(names)), body_jntnum=np.ones(len(names), dtype=int),
        jnt_type=np.zeros(len(names), dtype=int), jnt_qposadr=7*np.arange(len(names)),
        geom_bodyid=np.arange(len(names)),
    )
    data = SimpleNamespace(qpos=np.zeros(7*len(names)), xpos=np.zeros((len(names), 3)),
                           cvel=np.zeros((len(names), 6)), ncon=0, contact=[])
    harness.model, harness.data = model, data

    class Robot:
        dt = .001
        limits = np.tile([-1., 1.], (7, 1))

        def __init__(self, *, frame_skip):
            harness.robot_created = True
            self.model, self.data, self.frame_skip = model, data, frame_skip
            self.sim_time = 0.

        def reset(self, *, arm_home, closure):
            if harness.reset_error:
                raise harness.reset_error
            harness.reset_calls.append((arm_home.copy(), closure))
            data.qpos[2::7] = 4.75
            data.qpos[3::7] = 1.

        def state(self):
            return np.zeros(harness.state_dim, dtype=np.float32)

        def body_pose(self, name):
            return {'Pos':data.xpos[model.body(name).id]}

        def step(self, action):
            harness.commands.append(action.copy())
            self.sim_time += self.dt*self.frame_skip
            step = len(harness.commands)
            body = model.body('scene'+task_specs.target_spec(harness.target)['body_suffix']).id
            data.xpos[body, 2] = 4.75+harness.lift
            data.contact = []
            if step <= harness.grasp_frames:
                fingers = [len(names)-2] if harness.one_finger else [len(names)-2, len(names)-1]
                data.contact = [SimpleNamespace(geom1=body, geom2=finger) for finger in fingers]
            data.ncon = len(data.contact)
            data.cvel[:] = 0.
            if step == harness.unstable_at:
                data.cvel[body, 3] = .03

        def render(self, frame):
            pass

        def disconnect(self):
            harness.disconnected = True

    class RGB:
        def __init__(self, robot):
            pass

        def capture(self, folder, frame):
            return {'base_rgb':np.zeros((224, 224, 3), dtype=np.uint8),
                    'wrist_rgb':np.ones((224, 224, 3), dtype=np.uint8),
                    'object_position':data.xpos.copy(), 'expert_action':np.ones(8)}

        def close(self):
            harness.camera_closed = True

    class Policy:
        def __init__(self, **kwargs):
            pass

        def get_server_metadata(self):
            harness.metadata_calls += 1
            return harness.metadata

        def infer(self, observation):
            harness.observations.append(observation)
            if harness.infer_error:
                raise harness.infer_error
            return {'actions':harness.predictions.copy()}

    def validate_live_profile(model, *, profile):
        harness.live_profiles.append(profile)
        if harness.live_error:
            raise harness.live_error

    def sample_reset(seed, profile='legacy'):
        harness.sample_calls.append((seed, profile))
        return np.array([.018, -.019]), .24

    def no_expert(*args, **kwargs):
        pytest.fail('纯 VLA 推理禁止专家动作或物体真值 IK 回退')

    def install(name, **attributes):
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)

    geometry = dict(unique_body=lambda model, suffix: 'scene'+suffix,
                    box_bounds=lambda *args: (np.zeros(3), np.ones(3)),
                    placement_geometry=lambda *args: {'inside':True},
                    validate_live_profile=validate_live_profile)
    control = dict(FixedBaseRobot=Robot, home_joints=lambda robot: np.zeros(7))
    mj = dict(mjtJoint=SimpleNamespace(mjJNT_FREE=0),
              mj_forward=lambda model, data: data.xpos.__setitem__(slice(None), data.qpos.reshape(-1, 7)[:, :3]))
    install('mujoco', **mj)
    install('tools.medication_sorting.robot_control', **control, solve_site_ik=no_expert)
    install('tools.medication_sorting.run_expert', **geometry, execute=no_expert)
    install('tools.medication_sorting.rendered_rgb', RenderedRGB=RGB)
    install('tools.medication_sorting.collect_episodes', NativeRGB=RGB)
    install('openpi_client', __path__=[])
    install('openpi_client.websocket_client_policy', WebsocketClientPolicy=Policy)
    install(RANDOMIZATION_MODULE, sample_reset=sample_reset,
            PROFILES=randomization.PROFILES, profile_config=randomization.profile_config)
    # 也隔离尚未延迟导入的旧入口，避免红灯阶段触及真实运行环境。
    for name, value in {**geometry, **control, 'RenderedRGB':RGB, 'NativeRGB':RGB,
                        'mujoco':sys.modules['mujoco']}.items():
        monkeypatch.setattr(infer_policy, name, value, raising=False)
    monkeypatch.setattr(sys, 'path', sys.path.copy())

    def run(*extra, defaults=False, steps=24):
        argv = ['infer_policy', '--output', str(harness.output), '--steps', str(steps)]
        if not defaults:
            argv += ['--layout', str(harness.layout)]
        monkeypatch.setattr(sys, 'argv', argv+list(extra))
        return infer_policy.main()

    harness.run = run
    harness.report = lambda: json.loads((harness.output/'report.json').read_text())
    harness.trace = lambda: [json.loads(line) for line in (harness.output/'trace.jsonl').read_text().splitlines()]
    return harness


@pytest.mark.parametrize('target', task_specs.TARGETS)
@pytest.mark.parametrize('profile', task_specs.EXPERT_PROFILES)
def test_each_target_uses_its_spec_and_records_actual_provenance(runtime, target, profile):
    runtime.target = target
    spec = task_specs.target_spec(target, profile=profile)
    if profile != 'original':
        source = json.loads(scaled_layout.CAPTURE_LAYOUT.read_text())
        runtime.layout.write_text(json.dumps(scaled_layout.make_scaled_bottle_layout(source)))
    runtime.run('--target', target, '--profile', profile, '--randomization-profile', 'tabletop_v1')
    report = runtime.report()
    assert report['object_name'] == target
    assert report['task'] == spec['task']
    assert report['box'] == 'scene'+spec['body_suffix']
    assert report['basket'] == 'scene'+spec['basket_suffix']
    assert report['expert_profile'] == profile
    assert report['randomization_profile'] == 'tabletop_v1'
    assert report['randomization_config'] == randomization.profile_config('tabletop_v1')
    assert report['experiment_layout'] == str(runtime.layout.resolve())
    assert report['experiment_layout_sha256'] == hashlib.sha256(runtime.layout.read_bytes()).hexdigest()
    assert report['policy_metadata'] == runtime.metadata
    assert runtime.metadata_calls == 1
    assert runtime.live_profiles == [profile]
    assert runtime.sample_calls == [(413, 'tabletop_v1')]
    assert runtime.reset_calls[0][1] == spec['initial_closure']
    np.testing.assert_allclose(runtime.commands[0], infer_policy.limit_action(
        runtime.predictions[0], np.r_[np.zeros(7), spec['initial_closure']],
        np.tile([-1., 1.], (7, 1)), max_joint_step=.24, max_gripper_step=.32))
    assert report['success'] is True
    assert report['grasp_verified'] is True
    assert report['controller'] == 'pi05_checkpoint_policy'
    assert report['expert_actions_used'] is False
    assert runtime.disconnected and runtime.camera_closed


def test_new_expert_profile_is_read_dynamically_and_uses_spec_closure(runtime, monkeypatch):
    profile = 'future_tabletop_profile'
    original_spec = task_specs.target_spec

    def updated_spec(target, *, profile='original'):
        spec = original_spec(target, profile='original' if profile == 'future_tabletop_profile' else profile)
        if profile == 'future_tabletop_profile':
            spec['initial_closure'] = 0.
        return spec

    monkeypatch.setattr(task_specs, 'EXPERT_PROFILES', (*task_specs.EXPERT_PROFILES, profile))
    monkeypatch.setattr(task_specs, 'target_spec', updated_spec)
    source = json.loads(scaled_layout.CAPTURE_LAYOUT.read_text())
    runtime.layout.write_text(json.dumps(scaled_layout.make_scaled_bottle_layout(source)))
    runtime.run('--profile', profile)
    assert runtime.reset_calls[0][1] == 0.
    assert runtime.live_profiles == [profile]
    assert runtime.report()['expert_profile'] == profile


def test_cli_defaults_preserve_original_target_layout_and_legacy_reset(runtime):
    runtime.run(defaults=True)
    report = runtime.report()
    assert report['object_name'] == task_specs.DEFAULT_TARGET
    assert report['expert_profile'] == 'original'
    assert report['randomization_profile'] == 'legacy'
    assert report['experiment_layout_sha256'] == hashlib.sha256(scaled_layout.CAPTURE_LAYOUT.read_bytes()).hexdigest()
    assert runtime.sample_calls == [(413, 'legacy')]


def test_policy_receives_only_rgb_state_and_target_prompt(runtime):
    runtime.run()
    for observation in runtime.observations:
        assert set(observation) == {'state', 'prompt', 'base_rgb', 'wrist_rgb'}
        assert observation['state'].shape == (8,)
        assert observation['prompt'] == task_specs.target_spec(runtime.target)['task']
    assert all(command.shape == (8,) for command in runtime.commands)


def test_initial_height_is_copied_after_reset_and_forward(runtime):
    runtime.run()
    report = runtime.report()
    assert report['start_z'] == pytest.approx(4.75)
    assert report['initial']['start_z'] == pytest.approx(4.75)
    assert report['initial']['box_pos'] == pytest.approx([.018, -.019, 4.75])
    assert report['maximum_lift'] == pytest.approx(.07)
    np.testing.assert_allclose(report['reset_offset'], [.018, -.019])
    assert report['reset_yaw'] == .24


@pytest.mark.parametrize('extra', [('--target', 'unknown'), ('--profile', 'unknown'),
                                  ('--randomization-profile', 'unknown')])
def test_unknown_cli_choices_fail_before_output_or_robot(runtime, extra):
    with pytest.raises(SystemExit) as error:
        runtime.run(*extra)
    assert error.value.code == 2
    assert not runtime.robot_created
    assert not runtime.output.exists()


@pytest.mark.parametrize('layout_supplied', [False, True])
def test_scaled_profile_requires_matching_explicit_layout(runtime, layout_supplied):
    with pytest.raises(ValueError, match='显式|不一致'):
        runtime.run('--profile', 'scaled_bottles_70', defaults=not layout_supplied)
    assert not runtime.robot_created
    assert not runtime.output.exists()


def test_live_profile_failure_keeps_report_and_never_executes(runtime):
    runtime.live_error = ValueError('实际瓶径不匹配')
    with pytest.raises(ValueError, match='实际瓶径'):
        runtime.run()
    report = runtime.report()
    assert report['success'] is False
    assert '实际瓶径' in report['error']
    assert report['start_z'] is None
    assert report['initial'] is None
    assert not runtime.reset_calls and not runtime.commands
    assert runtime.disconnected


def test_reset_failure_does_not_mask_original_error_with_initial_height(runtime):
    runtime.reset_error = RuntimeError('初始重置失败')
    with pytest.raises(RuntimeError, match='初始重置失败'):
        runtime.run()
    report = runtime.report()
    assert report['start_z'] is None and report['initial'] is None
    assert '初始重置失败' in report['error']
    assert not runtime.commands and runtime.disconnected


def test_inference_error_never_uses_expert_fallback(runtime):
    runtime.infer_error = RuntimeError('模型推理失败')
    with pytest.raises(RuntimeError, match='模型推理失败'):
        runtime.run()
    assert not runtime.commands
    assert runtime.report()['expert_actions_used'] is False
    assert runtime.disconnected and runtime.camera_closed


@pytest.mark.parametrize('bad_actions', [np.zeros((4, 7)), np.zeros((0, 8)), np.full((4, 8), np.nan)])
def test_invalid_policy_actions_fail_without_execution(runtime, bad_actions):
    runtime.predictions = bad_actions
    with pytest.raises((RuntimeError, ValueError), match='动作'):
        runtime.run()
    assert not runtime.commands and runtime.disconnected


def test_invalid_state_is_rejected_before_policy_call(runtime):
    runtime.state_dim = 7
    with pytest.raises(ValueError, match='8 维'):
        runtime.run()
    assert not runtime.observations and not runtime.commands


@pytest.mark.parametrize('condition', ['one_finger', 'low_lift', 'brief_grasp', 'short_settle', 'unstable'])
def test_success_still_requires_grasp_lift_and_twenty_consecutive_settled_frames(runtime, condition):
    steps = 24
    if condition == 'one_finger':
        runtime.one_finger = True
    elif condition == 'low_lift':
        runtime.lift = .059
    elif condition == 'brief_grasp':
        runtime.grasp_frames = 2
    elif condition == 'short_settle':
        steps = 22
    else:
        runtime.unstable_at = 15
    with pytest.raises(SystemExit) as error:
        runtime.run(steps=steps)
    assert error.value.code == 1
    assert runtime.report()['success'] is False
    assert len(runtime.commands) == steps


def test_success_waits_until_twentieth_consecutive_released_settled_frame(runtime):
    runtime.run()
    trace = runtime.trace()
    assert len(trace) == 23
    assert trace[2]['grasp_streak'] == 3
    assert not trace[2]['released']
    assert all(row['inside'] and row['released'] and row['settled'] for row in trace[-20:])
