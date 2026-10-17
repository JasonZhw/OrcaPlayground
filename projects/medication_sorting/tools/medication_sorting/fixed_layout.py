"""创建实验 Layout 副本，只把 G1 根关节设为 fixed。"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROBOT = 'unitree_g1_1'
JOINT_COMPONENT = '{E544201F-83F6-419F-B3DD-1D6559FFDFB0}'
ROOT_ENTITY = [{'name': name, 'index': 0} for name in ('mujoco_wrap', 'rootPrim', 'pelvis', 'pelvis')]


def make_fixed_layout(source, *, robot_offset=(0, 0, 0),robot_yaw_degrees=0,
                      basket_scale_factor=1.):
    layout = copy.deepcopy(source)
    robots = []
    def walk(actors):
        for actor in actors:
            if actor['name'] == ROBOT:
                robots.append(actor)
            walk(actor.get('children', []))
    walk(layout['actors'])
    if len(robots) != 1:
        raise ValueError('必须包含唯一 G1 机器人')
    robot = robots[0]
    if not robot.get('asset_path', '').endswith('/g1_pick_with_gripper_usda'):
        raise ValueError('机器人资产与已验证的根关节路径不一致')
    offset = np.asarray(robot_offset, dtype=float)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError('机器人偏移必须为有限三维向量')
    robot['transform']['position'] = (np.asarray(robot['transform']['position']) + offset).tolist()
    if not np.isfinite(robot_yaw_degrees):
        raise ValueError('机器人朝向角必须有限')
    if robot_yaw_degrees:
        qw=robot['transform']['rotation']
        original_rotation=Rotation.from_quat([*qw[1:],qw[0]])
        new_rotation=Rotation.from_euler('z',robot_yaw_degrees,degrees=True)*original_rotation
        # 此版本 G1 资产 pelvis 相对于 Actor 原点的固定偏移，围绕基座旋转。
        pivot=np.array([-.32,0,.79])
        robot['transform']['position']=(np.asarray(robot['transform']['position'])+
            original_rotation.apply(pivot)-new_rotation.apply(pivot)).tolist()
        xyzw=new_rotation.as_quat()
        robot['transform']['rotation']=[xyzw[3],*xyzw[:3]]
    # Layout 中旧的 translate override 会覆盖新的顶层 transform，移除重叠项。
    if np.any(offset) or robot_yaw_degrees:
        for override in robot.get('entity_overrides', []):
            if not override['entity_path']:
                for group in override['group_overrids']:
                    group['property_overrides'] = [p for p in group['property_overrides']
                                                   if not p['name'].startswith('translate.')
                                                   and not (robot_yaw_degrees and p['name'].startswith('rotate.'))]
    overrides = robot.setdefault('entity_overrides', [])
    entity = next((o for o in overrides if o['entity_path'] == ROOT_ENTITY), None)
    if entity is None:
        entity = {'entity_path': copy.deepcopy(ROOT_ENTITY), 'group_overrids': []}
        overrides.append(entity)
    group = next((g for g in entity['group_overrids'] if g['type_id'] == JOINT_COMPONENT), None)
    if group is None:
        group = {'type_id': JOINT_COMPONENT, 'type_index': 0, 'property_overrides': []}
        entity['group_overrids'].append(group)
    group['property_overrides'] = [p for p in group['property_overrides'] if p['name'] != 'jointType']
    group['property_overrides'].append({'name': 'jointType', 'value': 'fixed', 'type': 'enum'})
    if not np.isfinite(basket_scale_factor) or basket_scale_factor <= 0:
        raise ValueError('筐缩放系数必须为有限正数')
    if basket_scale_factor != 1:
        def scale_basket(actors):
            for actor in actors:
                if actor['name'] == 'purple_plastic_crates_1':
                    actor['transform']['scale'] *= basket_scale_factor
                    for override in actor.get('entity_overrides', []):
                        if not override['entity_path']:
                            for group in override['group_overrids']:
                                for prop in group['property_overrides']:
                                    if prop['name'] == 'uniformScale':
                                        prop['value'] = actor['transform']['scale']
                scale_basket(actor.get('children', []))
        scale_basket(layout['actors'])
    return layout


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument('--source', type=Path, default=root/'data/medication_sorting/scenes/drag_classfication.json')
    parser.add_argument('--output', type=Path, default=root/'data/medication_sorting/scenes/drag_classfication_fixed_base.json')
    parser.add_argument('--robot-offset', type=float, nargs=3, default=(0, 0, 0))
    parser.add_argument('--robot-yaw',type=float,default=0)
    parser.add_argument('--basket-scale-factor',type=float,default=1.)
    args = parser.parse_args()
    result = make_fixed_layout(json.loads(args.source.read_text()), robot_offset=args.robot_offset,
                               robot_yaw_degrees=args.robot_yaw,basket_scale_factor=args.basket_scale_factor)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(args.output)
