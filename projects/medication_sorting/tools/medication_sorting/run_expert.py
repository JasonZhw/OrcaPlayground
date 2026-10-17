"""固定基座四种药品专家试验；默认兼容单药盒，失败报告也保留。"""

import argparse
from datetime import datetime
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from tools.medication_sorting.robot_control import (
    FixedBaseRobot, PREFIX, RIGHT_JOINTS, SITE, TOP_DOWN_ROTATION, solve_site_ik, home_joints,
    interpolate_actions,
)

from tools.medication_sorting.task_specs import DEFAULT_TARGET, TARGETS, EXPERT_PROFILES, target_spec

ROOT = Path(__file__).resolve().parents[2]


def grasp_rotation(target, *, profile='original', yaw=0., extra_tilt_x_degrees=0.):
    if not np.isfinite(extra_tilt_x_degrees) or abs(extra_tilt_x_degrees)>20:
        raise ValueError('额外抓取倾角必须在正负20度内')
    return (Rotation.from_euler('z', yaw).as_matrix()
            @ Rotation.from_euler('x', target_spec(target, profile=profile).get('tilt_x_degrees',0.)+extra_tilt_x_degrees,
                                  degrees=True).as_matrix() @ TOP_DOWN_ROTATION)


def placement_rotation(target, grasp, *, profile='original', align_grasp_to_object=False,
                       extra_grasp_tilt_x_degrees=0.):
    return grasp_rotation(target, profile=profile) if align_grasp_to_object or extra_grasp_tilt_x_degrees else grasp


def approach_target(center, height, offset=(0.,0.)):
    """只扰动药盒上方的接近航点；下探仍使用实际药盒中心。"""
    center=np.asarray(center,dtype=float);offset=np.asarray(offset,dtype=float)
    if (center.shape!=(2,) or offset.shape!=(2,)
        or not np.isfinite(np.r_[center,height,offset]).all()
        or np.any(np.abs(offset)>.03)):
        raise ValueError('接近偏移必须是有限二维值，每轴不超过3cm')
    return np.r_[center+offset,height]


def upright_gripper_rotation(hand_rotation, box_rotation):
    """保持夹持相对姿态，把药盒的长边和竖边转回世界坐标轴。"""
    return np.asarray(box_rotation).T @ np.asarray(hand_rotation)


def effective_placement_offset(requested, spec):
    """专家落点补偿和调用方偏移均使用筐的局部坐标。"""
    offsets = [np.asarray(value, dtype=float) for value in
               (requested, spec.get('placement_bias', (0., 0.)))]
    if any(value.shape != (2,) or not np.isfinite(value).all() for value in offsets):
        raise ValueError('落筐偏移及补偿必须是有限二维值')
    result = offsets[0] + offsets[1]
    if not np.isfinite(result).all():
        raise ValueError('合成落筐偏移必须有限')
    return result


def unique_body(model, suffix):
    names = [model.body(i).name for i in range(model.nbody) if model.body(i).name.endswith(suffix)]
    if len(names) != 1:
        raise RuntimeError(f'目标刚体不唯一：{suffix}: {names}')
    return names[0]


def validate_live_profile(model, *, profile):
    if profile not in ('scaled_bottles_70', 'randomized_tabletop_70'):
        return
    for suffix, diameter in [('_Medical_32', .063), ('_Medical_31', .056)]:
        body = model.body(unique_body(model, suffix)).id
        cylinders = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body
                     and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER
                     and (model.geom_contype[g] or model.geom_conaffinity[g])]
        actual = max((2*model.geom_size[g, 0] for g in cylinders), default=0.)
        if not np.isclose(actual, diameter, atol=1e-5, rtol=0):
            raise ValueError(f'实际瓶径与缩小配置不匹配：{suffix} {actual}m，期望{diameter}m')


def box_bounds(model, data, name, *, frame_name=None):
    """实际碰撞外形的精确轴向边界；兼容药瓶和子刚体，忽略纯视觉几何。"""
    bodies = {model.body(name).id}
    for b in range(model.nbody):
        if model.body_parentid[b] in bodies:
            bodies.add(b)
    geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] in bodies
             and (model.geom_contype[g] or model.geom_conaffinity[g])]
    origin, axes = np.zeros(3), np.eye(3)
    if frame_name is not None:
        frame = model.body(frame_name).id
        origin, axes = data.xpos[frame], data.xmat[frame].reshape(3,3)
    bounds = []
    for g in geoms:
        kind, size = model.geom_type[g], model.geom_size[g]
        rotation = axes.T @ data.geom_xmat[g].reshape(3,3)
        center = (data.geom_xpos[g]-origin) @ axes
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            extent = np.abs(rotation) @ size
        elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            extent = size[0]*np.sqrt(np.sum(rotation[:,:2]**2,axis=1))+size[1]*np.abs(rotation[:,2])
        elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            extent = size[0]+size[1]*np.abs(rotation[:,2])
        elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            extent = np.full(3,size[0])
        elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            extent = np.sqrt(np.sum((rotation*size)**2,axis=1))
        elif kind == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[g]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            vertices = model.mesh_vert[start:start+count] @ rotation.T + center
            bounds.extend([vertices.min(axis=0),vertices.max(axis=0)])
            continue
        else:
            raise RuntimeError(f'{name} 包含不支持的碰撞类型：{kind}')
        bounds.extend([center-extent,center+extent])
    if not bounds:
        raise RuntimeError(f'{name} 没有可识别的碰撞体')
    return np.min(bounds,axis=0),np.max(bounds,axis=0)


def placement_geometry(model,data,box,basket):
    """按筐的真实底板和当前姿态判定，兼容不同缩放和轻微物理漂移。"""
    body=model.body(basket).id
    floors=[g for g in np.flatnonzero(model.geom_bodyid==body)
            if model.geom_type[g]==mujoco.mjtGeom.mjGEOM_BOX
            and model.geom_size[g,2]<.2*min(model.geom_size[g,:2])]
    if not floors:
        raise RuntimeError('目标筐没有可识别的底板碰撞体')
    floor=max(floors,key=lambda g:np.prod(model.geom_size[g,:2]))
    pos,size=model.geom_pos[floor],model.geom_size[floor]
    low,high=box_bounds(model,data,box,frame_name=basket)
    height=float(pos[2]+size[2])
    inside=bool(np.all(low[:2]>pos[:2]-size[:2]) and np.all(high[:2]<pos[:2]+size[:2])
                and height-.003<=low[2]<height+.03)
    return {'inside':inside,'local_bounds':[low.tolist(),high.tolist()],
            'floor_half_size':size[:2].tolist(),'floor_height':height,
            'floor_center':pos.tolist()}


def execute(robot, output, *, observer=None, reset_offset=(0.,0.), reset_yaw=0.,
            approach_offset=(0.,0.), target=DEFAULT_TARGET, reset=True,
            placement_offset=(0.,0.), profile='original', align_grasp_to_object=False,
            diagnostics=False, extra_grasp_tilt_x_degrees=0., after_step=None, max_steps=None,
            upright_before_lower=False):
    spec = target_spec(target, profile=profile)
    if max_steps is not None and (type(max_steps) is not int or max_steps <= 0):
        raise ValueError('动作预算必须是正整数')
    placement_offset = effective_placement_offset(placement_offset, spec)
    if not reset and (np.any(np.asarray(reset_offset)) or reset_yaw):
        raise ValueError('连续分拣期间禁止重置物体位姿')
    approach_target([0.,0.],0.,approach_offset)
    validate_live_profile(robot.model, profile=profile)
    output.mkdir(parents=True,exist_ok=False)
    box = unique_body(robot.model,spec['body_suffix'])
    basket = unique_body(robot.model,spec['basket_suffix'])
    report = {'success':False,'physics_backend':'OrcaGymLocal / MuJoCo implicitfast',
              'box':box,'basket':basket,'phases':[],'lift_verified':False,
              'object_name':target,'task':spec['task'],'reset_at_start':reset,
              'expert_profile':profile,
              'expert_parameters':dict(spec),
              'object_mass_kg':float(robot.model.body(box).mass[0]),
              'object_inertia_kg_m2':robot.model.body(box).inertia.tolist(),
              'placement_offset':placement_offset.tolist(),
              'physics_diagnostics_enabled':bool(diagnostics),
              'upright_before_lower':bool(upright_before_lower),
              'no_object_teleport_during_episode':True}
    trace = []
    try:
        home = home_joints(robot) if reset else robot.state()[:7].copy()
        closure = spec['initial_closure'] if reset else float(robot.state()[7])
        if reset:
            robot.reset(arm_home=home,closure=closure)
        # 每回合开始时重置初始位置，采集过程只使用物理执行器。
        body_id=robot.model.body(box).id
        free_joint=robot.model.body_jntadr[body_id]
        if free_joint < 0 or robot.model.jnt_type[free_joint] != mujoco.mjtJoint.mjJNT_FREE:
            raise RuntimeError('目标物品必须是可被真实夹起的自由刚体')
        qadr=robot.model.jnt_qposadr[free_joint]
        if reset:
            robot.data.qpos[qadr:qadr+2] += np.asarray(reset_offset)
            xyzw=Rotation.from_euler('z',reset_yaw).as_quat()
            robot.data.qpos[qadr+3:qadr+7] = [xyzw[3],*xyzw[:3]]
            mujoco.mj_forward(robot.model,robot.data)
        report.update(reset_offset=list(reset_offset),reset_yaw=float(reset_yaw),
                      approach_offset=list(approach_offset),
                      align_grasp_to_object=align_grasp_to_object)
        q = home.copy()
        target_orientation=grasp_rotation(target, profile=profile, extra_tilt_x_degrees=extra_grasp_tilt_x_degrees)
        report['extra_grasp_tilt_x_degrees']=float(extra_grasp_tilt_x_degrees)
        frame = 0
        previous_action=np.r_[q,closure]

        def contact_sides():
            sides = set()
            for c in robot.data.contact[:robot.data.ncon]:
                names = [robot.model.body(robot.model.geom_bodyid[g]).name for g in (c.geom1,c.geom2)]
                if box not in names:
                    continue
                for branch in ('inner','outer'):
                    if any('gripper_r_'+branch+'_link4' in n for n in names):
                        sides.add(branch)
            return sorted(sides)

        def tick(action,phase):
            nonlocal frame,previous_action
            for sub_action in interpolate_actions(previous_action,action,robot.limits,
                                                  robot.dt*robot.frame_skip):
                sample(sub_action,phase)
                previous_action=sub_action.copy()

        def sample(action,phase):
            nonlocal frame
            if max_steps is not None and frame >= max_steps:
                raise RuntimeError('单目标动作预算已耗尽')
            # 观测位于动作之前；两幅图像必须来自同一个暂停的物理时刻。
            if observer is not None:
                observer(robot,action,phase,frame)
            robot.step(action)
            record = {'time':robot.sim_time,'phase':phase,'ee':robot.site_pose()['xpos'].tolist(),
                      'state':robot.state().tolist(),'action':np.asarray(action).tolist(),
                      'box_pos':robot.body_pose(box)['Pos'].tolist(),
                      'box_quat':robot.body_pose(box)['Quat'].tolist(),'contact_sides':contact_sides()}
            if diagnostics:
                from tools.medication_sorting.physics_diagnostics import capture_snapshot
                record['physics'] = capture_snapshot(robot.model, robot.data, target=box,
                    basket=basket, site=SITE, joint_names=RIGHT_JOINTS)
            trace.append(record)
            frame += 1
            if after_step is not None:
                after_step(robot, record)

        def move(position,seconds,phase,new_closure=None,rotation=None,carry_box=False):
            nonlocal q,closure
            start = robot.site_pose()['xpos'].copy()
            start_rotation = robot.site_pose()['xmat'].copy()
            rotations = Slerp([0,1],Rotation.from_matrix([start_rotation,
                             target_orientation if rotation is None else rotation]))
            if carry_box:
                bounds=box_bounds(robot.model,robot.data,box)
                start[:2]=(bounds[0][:2]+bounds[1][:2])/2
            c0 = closure
            c1 = closure if new_closure is None else new_closure
            n = round(seconds/(robot.dt*robot.frame_skip))
            for i in range(n):
                alpha=(i+1)/n
                blend=alpha*alpha*(3-2*alpha)
                target = start+(np.asarray(position)-start)*blend
                if carry_box:
                    bounds=box_bounds(robot.model,robot.data,box)
                    actual_center=(bounds[0][:2]+bounds[1][:2])/2
                    target[:2] -= actual_center-robot.site_pose()['xpos'][:2]
                seed=robot.data.qpos.copy()
                seed[robot.arm_qpos] = q
                target_rotation=rotations(blend).as_matrix()
                try:
                    try:
                        solution = solve_site_ik(robot.model,seed,RIGHT_JOINTS,SITE,target,target_rotation,
                                          attempts=1,position_tolerance=.008,rotation_tolerance=.09,
                                          joint_regularization=.001)
                    except RuntimeError:
                        if 'bottle' not in report['object_name']:
                            raise
                        solution = solve_site_ik(robot.model,seed,RIGHT_JOINTS,SITE,target,target_rotation,
                                          attempts=8,position_tolerance=.008,rotation_tolerance=.09,
                                          joint_regularization=.001)
                    q = solution['joints']
                except RuntimeError:
                    report['failed_target']={'phase':phase,'position':target.tolist(),
                        'rotation':target_rotation.tolist(),'state':robot.state().tolist()}
                    if diagnostics:
                        from tools.medication_sorting.physics_diagnostics import capture_snapshot
                        report['failed_target']['ik_seed_qpos'] = seed.tolist()
                        report['failed_target']['physics'] = capture_snapshot(robot.model, robot.data,
                            target=box, basket=basket, site=SITE, joint_names=RIGHT_JOINTS)
                    raise
                closure=c0+(c1-c0)*blend
                tick(np.r_[q,closure],phase)
            robot.render(frame)
            report['phases'].append(trace[-1])
            (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
            print(phase,{key:value for key,value in trace[-1].items() if key!='physics'},flush=True)

        for _ in range(round(.5/(robot.dt*robot.frame_skip))):tick(np.r_[q,closure],'settle_initial')
        box_start = robot.body_pose(box)['Pos'].copy()
        if align_grasp_to_object:
            rotation = robot.data.xmat[body_id].reshape(3,3)
            yaw = float(np.arctan2(rotation[1,0], rotation[0,0]))
            target_orientation = grasp_rotation(target, profile=profile, yaw=yaw,
                                                 extra_tilt_x_degrees=extra_grasp_tilt_x_degrees)
            report['grasp_yaw_rad'] = yaw
        low,high=box_bounds(robot.model,robot.data,box)
        report['initial_box_bounds']=[low.tolist(),high.tolist()]
        grasp_xy = (low[:2]+high[:2])/2
        grasp_z = high[2]+spec.get('grasp_clearance',-.005)
        travel_z=spec['travel_z']
        move(approach_target(grasp_xy,min(high[2]+.14,travel_z),approach_offset),2.,'approach',
             new_closure=spec['initial_closure'])
        move([*grasp_xy,grasp_z],2.,'descend')
        move([*grasp_xy,grasp_z],1.5,'close',new_closure=1.)
        if contact_sides() != ['inner','outer']:
            raise RuntimeError('闭合后没有获得目标物品与两指的同时接触')
        move([*grasp_xy,travel_z],2.,'lift')
        report['lift_height'] = float(robot.body_pose(box)['Pos'][2]-box_start[2])
        report['lift_verified'] = report['lift_height'] > .06
        if not report['lift_verified']:
            raise RuntimeError('目标物品没有被真实抬起至少 6cm')
        basket_pos = robot.body_pose(basket)['Pos']
        basket_rotation=robot.data.xmat[robot.model.body(basket).id].reshape(3,3)
        destination = (basket_pos + basket_rotation @ np.r_[placement_offset,0.])[:2]
        # 药盒在夹持中会倾斜；专家用真实物体中心反馈调整落点和手腕，
        # 只改变执行器目标，不改物体位置或添加吸附约束。
        place_rotation = placement_rotation(target, target_orientation, profile=profile,
                                             align_grasp_to_object=align_grasp_to_object,
                                             extra_grasp_tilt_x_degrees=extra_grasp_tilt_x_degrees)
        move([*destination,travel_z],3.,'carry',rotation=place_rotation,carry_box=True)
        if upright_before_lower:
            place_rotation = upright_gripper_rotation(robot.site_pose()['xmat'],
                                robot.data.xmat[body_id].reshape(3,3))
            report['upright_target_rotation'] = place_rotation.tolist()
            move([*destination,travel_z],1.5,'upright',rotation=place_rotation,carry_box=True)
        move([*destination,spec.get('lower_z',4.99)],1.5,'lower',rotation=place_rotation,carry_box=True)
        release_position=robot.site_pose()['xpos'].copy()
        move(release_position,1.5,'release',new_closure=0.,rotation=place_rotation)
        move([*destination,travel_z],1.5,'retreat')
        for _ in range(round(3./(robot.dt*robot.frame_skip))):tick(np.r_[q,closure],'settle_final')
        robot.render(frame)
        low,high=box_bounds(robot.model,robot.data,box)
        center=(low+high)/2
        placement=placement_geometry(robot.model,robot.data,box,basket)
        inside=placement['inside']
        speed = np.linalg.norm(robot.data.cvel[body_id,3:])
        released = not contact_sides()
        success = bool(inside and released and speed<.02)
        report.update(success=success,final_box_bounds=[low.tolist(),high.tolist()],
                      final_box_center=center.tolist(),frames=frame,final_linear_speed=float(speed),
                      released=released,placement=placement)
        if not success:
            raise RuntimeError('松爪后物品未完整落入目标筐或尚未静止')
        return report
    except Exception as exc:
        report['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
        (output/'trace.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in trace))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'data/medication_sorting/experiments'/datetime.now().strftime('%Y%m%d-%H%M%S'))
    parser.add_argument('--target',choices=TARGETS,default=DEFAULT_TARGET)
    parser.add_argument('--profile',choices=EXPERT_PROFILES,default='original')
    args=parser.parse_args()
    robot=FixedBaseRobot()
    try:
        print(json.dumps(execute(robot,args.output,target=args.target,profile=args.profile),ensure_ascii=False,indent=2))
    finally:
        robot.disconnect()


if __name__=='__main__':
    main()
