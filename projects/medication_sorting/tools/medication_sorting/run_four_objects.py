"""逐物品试抓/双 RGB 采集；连续模式不在物品之间重置场景。"""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.medication_sorting.collect_episodes import collect
from tools.medication_sorting.rendered_rgb import RenderedRGB
from tools.medication_sorting.robot_control import FixedBaseRobot
from tools.medication_sorting.run_expert import TARGETS, execute, placement_geometry, unique_body
from tools.medication_sorting.verify_episodes import verify_episode
from tools.medication_sorting.task_specs import EXPERT_PROFILES
from tools.medication_sorting.scaled_layout import resolve_experiment_layout

ROOT=Path(__file__).resolve().parents[2]
LAYOUT=ROOT/'data/medication_sorting/scenes/drag_classfication_capture.json'
DROP_OFFSETS={
    'pharyngitis_tablets_box_1':(-.035,0.),
    'fu_shi_gao_herb_drink_packaging_box_1':(.018,0.),
    'melatonin_tablet_bottle_1':(0.,-.048),
    'calcium_gluconate_tablets_bottle_1':(0.,.048),
}


def summarize_results(results, *, rgb, continuous=False, final=None):
    counts={name:{'attempts':0,'physical_successes':0,'accepted_episodes':0} for name in TARGETS}
    for row in results:
        count=counts[row['target']]
        count['attempts']+=1
        count['physical_successes']+=int(row['success'])
        count['accepted_episodes']+=int(rgb and row['success'] and row.get('quality_accepted') is True)
    physical=all(v['physical_successes']>0 for v in counts.values())
    accepted=all(v['accepted_episodes']>0 for v in counts.values())
    final=final or {}
    full_scene=set(final)==set(TARGETS) and all(v.get('inside') and v.get('stable') for v in final.values())
    continuous_success=bool(continuous and len(results)==4
                            and all(v['success'] for v in results) and full_scene)
    success=continuous_success if continuous else physical
    if rgb:success=success and accepted
    return {'success':success,'by_object':counts,'all_four_objects_physically_succeeded':physical,
            'all_four_objects_have_accepted_data':accepted,
            'continuous_sorting_success':continuous_success,
            'final_scene':final,'results':results,
            'training_started':False}


def final_scene_status(robot):
    status={}
    for target,spec in TARGETS.items():
        body=unique_body(robot.model,spec['body_suffix'])
        basket=unique_body(robot.model,spec['basket_suffix'])
        geometry=placement_geometry(robot.model,robot.data,body,basket)
        speed=float(np.linalg.norm(robot.data.cvel[robot.model.body(body).id,3:]))
        status[target]={'inside':geometry['inside'],'stable':speed<.02,
                        'linear_speed':speed,'placement':geometry}
    return status


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rgb',action='store_true',help='保存真实同步双视角、状态、动作和逐帧证据')
    parser.add_argument('--profile',choices=EXPERT_PROFILES,default='original')
    parser.add_argument('--continuous',action='store_true',help='首物品后禁止reset，失败即停止该次连续整理')
    parser.add_argument('--targets',nargs='+',choices=TARGETS,default=list(reversed(TARGETS)))
    parser.add_argument('--layout',type=Path,help='必须与当前已加载的场景相同；缩小配置必须显式提供')
    parser.add_argument('--output',type=Path,default=ROOT/'data/medication_sorting/multi_object'/datetime.now().strftime('%Y%m%d-%H%M%S'))
    args=parser.parse_args()
    args.layout=resolve_experiment_layout(args.layout, profile=args.profile)
    if len(set(args.targets))!=len(args.targets):
        parser.error('物品列表不能重复')
    layout_hash=hashlib.sha256(args.layout.read_bytes()).hexdigest()
    args.output.mkdir(parents=True,exist_ok=False)
    robot=FixedBaseRobot(frame_skip=200)
    rgb=None
    results=[]
    final={}
    manifest={'mode':'continuous' if args.continuous else 'independent_reset_per_object',
              'expert_profile':args.profile,
              'rgb':args.rgb,'layout':str(args.layout.resolve()),'layout_sha256':layout_hash,
              'targets':args.targets,'source_model_xml_sha256':hashlib.sha256(Path(robot.xml_path).read_bytes()).hexdigest()}
    (args.output/'session.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    try:
        if args.rgb:rgb=RenderedRGB(robot)
        for index,target in enumerate(args.targets):
            reset=not args.continuous or index==0
            offset=DROP_OFFSETS[target] if args.continuous else (0.,0.)
            folder=args.output/target/'episode_0000'
            folder.parent.mkdir(parents=True,exist_ok=True)
            row={'target':target,'success':False,'quality_accepted':False,'path':str(folder.resolve()),
                 'reset_at_start':reset,'placement_offset':offset}
            print('开始物品',target,'reset=',reset,flush=True)
            try:
                if rgb is None:
                    report=execute(robot,folder,target=target,reset=reset,placement_offset=offset,profile=args.profile)
                else:
                    report=collect(robot,rgb,folder,target=target,layout=args.layout,
                                   reset=reset,placement_offset=offset,profile=args.profile)
                row['success']=report['success']
            except Exception as exc:
                row['error']=f'{type(exc).__name__}: {exc}'
            if args.rgb and (folder/'metadata.json').is_file():
                quality=verify_episode(folder)
                (folder/'quality.json').write_text(json.dumps(quality,ensure_ascii=False,indent=2))
                row['quality_accepted']=quality['accepted']
                row['quality_errors']=quality['errors']
            results.append(row)
            summary=summarize_results(results,rgb=args.rgb,continuous=args.continuous)
            (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
            print('物品结果',json.dumps(row,ensure_ascii=False),flush=True)
            if args.continuous and not row['success']:break
        if args.continuous:final=final_scene_status(robot)
        summary=summarize_results(results,rgb=args.rgb,continuous=args.continuous,final=final)
        (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    finally:
        if rgb is not None:rgb.close()
        robot.disconnect()
    print('报告',args.output/'summary.json',flush=True)
    if not summary['success']:raise SystemExit(1)


if __name__=='__main__':main()
