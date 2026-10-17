"""四药品纯 VLA 推理：动作仅来自双 RGB、G1 的 8 维 state 和任务 prompt。

物体真值仅用于回合初始化、轨迹记录与验收，不生成动作或提供专家回退。
导入及 --help 不加载机器人、相机或策略服务依赖。
"""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from tools.medication_sorting import task_specs
from tools.medication_sorting.pi05_adapter import CAPTURE_CONFIG
from tools.medication_sorting.scaled_layout import resolve_experiment_layout

ROOT=Path(__file__).resolve().parents[2]


def bounded_chunk(actions,*,execute_chunk,remaining):
    if execute_chunk<=0 or remaining<=0:
        raise ValueError('执行长度与剩余步数必须为正')
    return actions[:min(execute_chunk,remaining)]


def reset_randomization(seed,profile='legacy'):
    """保留旧调用默认值，实际分布与采集共用唯一采样入口。"""
    from tools.medication_sorting.randomization import sample_reset
    return sample_reset(seed,profile=profile)


def finger_contacts(model,data,box):
    touched=set()
    for contact in data.contact[:data.ncon]:
        names=[model.body(model.geom_bodyid[g]).name for g in (contact.geom1,contact.geom2)]
        if box in names:
            for side in ('inner','outer'):
                if any('gripper_r_'+side+'_link4' in name for name in names):touched.add(side)
    return touched


def free_body_qpos_address(model,name):
    import mujoco

    body=model.body(name).id
    joint=model.body_jntadr[body]
    if model.body_jntnum[body]!=1 or joint<0 or model.jnt_type[joint]!=mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError('药盒必须是单个自由关节刚体，禁止写入非自由关节 qpos')
    return int(model.jnt_qposadr[joint])


def grasp_streak(previous,contacts,height):
    return previous+1 if contacts=={'inner','outer'} and height>.06 else 0


def validate_policy_metadata(metadata,*,control_hz=5,camera_backend='viewport'):
    expected={'robot_type':'unitree_g1_fixed_base_right_arm','state_dim':8,'action_dim':8,
              'joint_units':'rad','joint_action_semantics':'absolute_position',
              'gripper_semantics':{'open':0.,'closed':1.},'observation_config':CAPTURE_CONFIG}
    if not isinstance(metadata,dict) or any(metadata.get(key)!=value for key,value in expected.items()):
        raise ValueError('策略服务的机器人、维度或动作语义与 G1 采集契约不符')
    if control_hz!=CAPTURE_CONFIG['control_hz'] or camera_backend!=CAPTURE_CONFIG['camera_backend']:
        raise ValueError('执行频率和相机后端必须与当前训练数据一致：5 Hz、viewport')


def limit_action(action,previous,limits, *, max_joint_step=.06,max_gripper_step=.08):
    proposed=np.asarray(action,dtype=float)
    previous=np.asarray(previous,dtype=float)
    limits=np.asarray(limits,dtype=float)
    if proposed.shape!=(8,) or not np.isfinite(proposed).all():
        raise ValueError('VLA 服务必须返回有限的 8 维动作')
    if (limits.shape!=(7,2) or not np.isfinite(limits).all()
        or np.any(limits[:,0]>limits[:,1])):
        raise ValueError('关节限位必须为有效的 7×2 区间')
    if (previous.shape!=(8,) or not np.isfinite(previous).all()
        or np.any(previous[:7]<limits[:,0]) or np.any(previous[:7]>limits[:,1])
        or not 0<=previous[7]<=1):
        raise ValueError('上一条命令必须有限且位于关节和夹爪限位内')
    if (not np.isfinite(max_joint_step) or not np.isfinite(max_gripper_step)
        or max_joint_step<0 or max_gripper_step<0):
        raise ValueError('单步速度限制必须为有限非负数')
    result=proposed.copy()
    result[:7]=np.clip(result[:7],limits[:,0],limits[:,1])
    result[7]=np.clip(result[7],0,1)
    result[:7]=np.clip(result[:7],previous[:7]-max_joint_step,previous[:7]+max_joint_step)
    result[7]=np.clip(result[7],previous[7]-max_gripper_step,previous[7]+max_gripper_step)
    return result


def main():
    from tools.medication_sorting.randomization import PROFILES, profile_config

    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=8000)
    parser.add_argument('--steps',type=int,default=600)
    parser.add_argument('--execute-chunk',type=int,default=4)
    parser.add_argument('--seed',type=int,default=413)
    parser.add_argument('--target',choices=task_specs.TARGETS,default=task_specs.DEFAULT_TARGET)
    parser.add_argument('--profile',choices=task_specs.EXPERT_PROFILES,default='original')
    parser.add_argument('--layout',type=Path,help='非原尺寸配置必须显式指定匹配的实验场景')
    parser.add_argument('--randomization-profile',choices=PROFILES,default='legacy')
    parser.add_argument('--camera-backend',choices=['viewport','sensor'],default='viewport')
    parser.add_argument('--control-hz',type=int,choices=[5,10,20],default=5)
    parser.add_argument('--output',type=Path,default=ROOT/'data/medication_sorting/inference'/datetime.now().strftime('%Y%m%d-%H%M%S'))
    args=parser.parse_args()
    if args.steps<=0 or args.execute_chunk<=0:
        parser.error('--steps 和 --execute-chunk 必须为正整数')
    spec=task_specs.target_spec(args.target,profile=args.profile)
    layout=resolve_experiment_layout(args.layout,profile=args.profile)
    layout_hash=hashlib.sha256(layout.read_bytes()).hexdigest()
    offset,yaw=reset_randomization(args.seed,profile=args.randomization_profile)
    sys.path.insert(0,str(ROOT/'software/openpi/packages/openpi-client/src'))
    args.output.mkdir(parents=True,exist_ok=False)
    robot=None
    rgb=None
    trace=[]
    report={'success':False,'grasp_verified':False,'controller':'pi05_checkpoint_policy','expert_actions_used':False,
            'seed':args.seed,'policy_metadata':None,
            'object_name':args.target,'task':spec['task'],'basket':None,
            'expert_profile':args.profile,'randomization_profile':args.randomization_profile,
            'randomization_config':profile_config(args.randomization_profile),
            'experiment_layout':str(layout.resolve()),'experiment_layout_sha256':layout_hash,
            'reset_offset':offset.tolist(),'reset_yaw':yaw,'initial':None,'start_z':None}
    try:
        from openpi_client.websocket_client_policy import WebsocketClientPolicy

        policy=WebsocketClientPolicy(host=args.host,port=args.port)
        report['policy_metadata']=policy.get_server_metadata()
        validate_policy_metadata(report['policy_metadata'],control_hz=args.control_hz,camera_backend=args.camera_backend)

        import mujoco
        from tools.medication_sorting.robot_control import FixedBaseRobot, home_joints
        from tools.medication_sorting.run_expert import (
            box_bounds, unique_body, placement_geometry, validate_live_profile,
        )

        robot=FixedBaseRobot(frame_skip=round(1000/args.control_hz))
        validate_live_profile(robot.model,profile=args.profile)
        box=unique_body(robot.model,spec['body_suffix'])
        basket=unique_body(robot.model,spec['basket_suffix'])
        report.update(box=box,basket=basket)
        home=home_joints(robot)
        closure=spec['initial_closure']
        robot.reset(arm_home=home,closure=closure)
        bid=robot.model.body(box).id
        qadr=free_body_qpos_address(robot.model,box)
        # 仅初始化允许改写目标自由关节；执行期间只发送策略动作。
        robot.data.qpos[qadr:qadr+2]+=offset
        from scipy.spatial.transform import Rotation
        changed=Rotation.from_euler('z',yaw).as_quat()
        robot.data.qpos[qadr+3:qadr+7]=changed[[3,0,1,2]]
        mujoco.mj_forward(robot.model,robot.data)
        initial_position=np.asarray(robot.body_pose(box)['Pos']).copy()
        start_z=float(initial_position[2])
        report['start_z']=start_z
        report['initial']={'box_pos':initial_position.tolist(),'start_z':start_z}
        previous=np.r_[home,closure]
        if args.camera_backend=='viewport':
            from tools.medication_sorting.rendered_rgb import RenderedRGB
            rgb=RenderedRGB(robot)
        else:
            from tools.medication_sorting.collect_episodes import NativeRGB
            rgb=NativeRGB(robot)
        frame=0
        maximum_lift=0.
        grasp_count=0
        while frame<args.steps:
            images=rgb.capture(args.output/'frames'/f'{frame:06d}',frame)
            state=np.asarray(robot.state(),dtype=np.float32)
            if state.shape!=(8,) or not np.isfinite(state).all():
                raise ValueError('G1 state 必须为有限的 8 维向量')
            observation={'state':state,'prompt':spec['task'],
                         'base_rgb':images['base_rgb'],'wrist_rgb':images['wrist_rgb']}
            result=policy.infer(observation)
            actions=np.asarray(result.get('actions'))
            if actions.ndim!=2 or actions.shape[1]!=8 or not len(actions):
                raise RuntimeError('推理服务动作 chunk 形状错误')
            for prediction in bounded_chunk(actions,execute_chunk=args.execute_chunk,remaining=args.steps-frame):
                period=robot.dt*robot.frame_skip
                action=limit_action(prediction,previous,robot.limits,
                                    max_joint_step=1.2*period,max_gripper_step=1.6*period)
                robot.step(action)
                previous=action
                position=robot.body_pose(box)['Pos']
                maximum_lift=max(maximum_lift,float(position[2]-start_z))
                contacts=finger_contacts(robot.model,robot.data,box)
                grasp_count=grasp_streak(grasp_count,contacts,float(position[2]-start_z))
                if grasp_count>=3:
                    report['grasp_verified']=True
                low,high=box_bounds(robot.model,robot.data,box)
                released=not any(box in [robot.model.body(robot.model.geom_bodyid[g]).name for g in (c.geom1,c.geom2)]
                    and any('gripper_r' in robot.model.body(robot.model.geom_bodyid[g]).name for g in (c.geom1,c.geom2))
                    for c in robot.data.contact[:robot.data.ncon])
                inside=placement_geometry(robot.model,robot.data,box,basket)['inside']
                settled=np.linalg.norm(robot.data.cvel[bid,3:])<.02
                trace.append({'frame':frame,'time':robot.sim_time,'prediction':prediction.tolist(),
                              'action':action.tolist(),'box_pos':position.tolist(),
                              'state':robot.state().tolist(),'contact_sides':sorted(contacts),
                              'grasp_streak':grasp_count,
                              'inside':bool(inside),'released':bool(released),'settled':bool(settled)})
                frame+=1
                if report['grasp_verified'] and maximum_lift>.06 and inside and released and settled:
                    recent=trace[-20:]
                    if len(recent)==20 and all(t['inside'] and t['released'] and t['settled'] for t in recent):
                        report['success']=True
                        break
            if report['success']:break
            if frame%40==0:print('VLA 推理',frame,'抬升',maximum_lift,flush=True)
        robot.render(frame)
        report.update(frames=frame,maximum_lift=maximum_lift,final_box_bounds=[low.tolist(),high.tolist()])
    except Exception as exc:
        report['error']=repr(exc)
        raise
    finally:
        try:
            (args.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str))
            (args.output/'trace.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in trace))
        finally:
            try:
                if rgb is not None and hasattr(rgb,'close'):rgb.close()
            finally:
                if robot is not None:robot.disconnect()
    print(json.dumps(report,ensure_ascii=False,indent=2,default=str))
    if not report['success']:raise SystemExit(1)


if __name__=='__main__':main()
