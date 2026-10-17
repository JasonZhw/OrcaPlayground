"""采集固定基座 G1 的真实物理示范与 OrcaLab 原生双 RGB 快照。"""

import argparse
import asyncio
import hashlib
import importlib.metadata
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from tools.medication_sorting.robot_control import FixedBaseRobot, PREFIX, RIGHT_JOINTS
from tools.medication_sorting.run_expert import (
    execute, TARGETS, DEFAULT_TARGET, validate_live_profile, effective_placement_offset,
)
from tools.medication_sorting.rendered_rgb import RenderedRGB
from tools.medication_sorting.task_specs import EXPERT_PROFILES, target_spec
from tools.medication_sorting.scaled_layout import resolve_experiment_layout
from tools.medication_sorting.randomization import PROFILES, profile_config, sample_reset

ROOT=Path(__file__).resolve().parents[2]
CAMERAS={'base_rgb':'head_cam','wrist_rgb':'camera_right'}


def exposure(rgb):
    """固定 gamma 提亮，训练与部署使用同一处理，不使用场景真值修改图像。"""
    return np.rint(255*(rgb.astype(np.float32)/255)**.45).astype(np.uint8)


class NativeRGB:
    def __init__(self,robot):
        self.robot=robot
        from orca_gym.protos import mjc_message_pb2 as pb
        self.pb=pb
        self.stub=robot.gym.stub
        for name,port in [('head_cam',7170),('camera_right',7172)]:
            self.call(self.stub.SetStreamingEnabled(pb.SetStreamingEnabledRequest(camera_name=name,enabled=False)))
            result=self.call(self.stub.SetCameraProperties(pb.SetCameraPropertiesRequest(camera_name=name,
                property=pb.CameraProperty(capture_rgb=True,width=224,height=224,near_clip=.15,
                                          vertical_fov=60 if name=='head_cam' else 90,
                                          color_port=port,depth_port=port+1,use_nvenc=False))))
            if result.status != 0:
                raise RuntimeError(f'相机配置失败：{result.error_message}')
            result=self.call(self.stub.SetStreamingEnabled(pb.SetStreamingEnabledRequest(camera_name=name,enabled=True)))
            if result.status != 0:
                raise RuntimeError(f'相机启动失败：{result.error_message}')

    def call(self,operation):
        return self.robot.loop.run_until_complete(asyncio.wait_for(operation,timeout=20))

    def capture(self,folder,index):
        folder.mkdir(parents=True,exist_ok=False)
        self.robot.render(index)
        # 仿真暂停，等待引擎绘制此状态；请求期间不推进物理。
        self.robot.loop.run_until_complete(asyncio.sleep(.08))
        response=self.call(self.stub.GetCameraFramePNG(self.pb.GetCameraFramePNGRequest(image_path=str(folder.resolve()))))
        transforms={item.name:item for item in response.name_transform}
        if not set(CAMERAS.values())<=transforms.keys():
            raise RuntimeError('双相机原生快照缺失')
        for camera,parts in [('head_cam',['head_camera1','head_camera2']),
                              ('camera_right',['camera_right_body1','camera_right_body2'])]:
            expected=[self.robot.body_pose(PREFIX+n)['Pos'] for n in parts]
            if min(np.linalg.norm(transforms[camera].pos-pos) for pos in expected)>.003:
                raise RuntimeError(f'{camera} 渲染位姿与当前物理状态不同步')
        deadline=time.monotonic()+5
        images={}
        while len(images)<2:
            for key,name in CAMERAS.items():
                files=list((folder/'color').glob(name+'_color_*.png'))
                if len(files)!=1:
                    continue
                try:
                    with Image.open(files[0]) as image:
                        image.load()
                        rgb=np.asarray(image.convert('RGB'))
                    if rgb.shape!=(224,224,3) or rgb.std()<1:
                        raise RuntimeError(f'{name} 原生 RGB 为空白或尺寸不符')
                    images[key]=exposure(rgb)
                except OSError:
                    pass
            if time.monotonic()>deadline:
                raise RuntimeError('原生双 RGB 快照没有及时落盘')
            if len(images)<2:
                self.robot.loop.run_until_complete(asyncio.sleep(.01))
        return images


def collect(robot,rgb,output, *, seed=0,randomize=False,approach_jitter=0.,
            target=DEFAULT_TARGET, layout=None, reset=True, placement_offset=(0.,0.),
            profile='original',randomization_profile='legacy'):
    expert_parameters=target_spec(target, profile=profile)
    actual_placement_offset=effective_placement_offset(placement_offset, expert_parameters)
    implementation={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                    for name in ('run_expert.py','robot_control.py','task_specs.py')}
    randomization_config=profile_config(randomization_profile)
    layout=resolve_experiment_layout(layout, profile=profile)
    layout_hash=hashlib.sha256(layout.read_bytes()).hexdigest()
    if not reset and randomize:
        raise ValueError('连续分拣期间禁止随机重置物体')
    if not np.isfinite(approach_jitter) or not 0<=approach_jitter<=.03:
        raise ValueError('接近航点扰动范围必须在0–0.03m')
    validate_live_profile(robot.model, profile=profile)
    rows=[]
    def observe(r,action,phase,index):
        images=rgb.capture(output/'frames'/f'{index:06d}',index)
        rows.append({'state':r.state(),'action':np.asarray(action,dtype=np.float32),
                     'sim_time':r.sim_time,'phase':phase,**images})
        if index%50==0:
            print('采集帧',index,'仿真时间',r.sim_time,flush=True)
    success=False
    try:
        rng=np.random.default_rng(seed)
        offset,yaw=sample_reset(seed,profile=randomization_profile) if randomize else (np.zeros(2),0.)
        if randomize:
            rng.uniform(size=3)  # 保留历史接近航点扰动使用的后续随机数。
        approach_offset=rng.uniform(-approach_jitter,approach_jitter,2)
        report=execute(robot,output,observer=observe,reset_offset=offset,reset_yaw=yaw,
                       approach_offset=approach_offset,target=target,reset=reset,
                       placement_offset=placement_offset,profile=profile,
                       align_grasp_to_object=randomize and randomization_profile!='legacy')
        success=report['success']
        return report
    finally:
        if rows:
            np.savez_compressed(output/'episode.npz',**{
                key:np.stack([row[key] for row in rows])
                for key in ('state','action','base_rgb','wrist_rgb','sim_time','phase')})
        camera_names=getattr(rgb,'camera_names',CAMERAS)
        alignment=getattr(rgb,'alignment','paused_physics_snapshot_with_camera_pose_check')
        metadata={'success':success,'complete':success,'synchronized':True,
                  'task':TARGETS[target]['task'],'joint_names':RIGHT_JOINTS,
                  'object_name':target,'basket_suffix':TARGETS[target]['basket_suffix'],
                  'expert_profile':profile,
                  'expert_parameters':expert_parameters,
                  'expert_implementation_sha256':implementation,
                  'reset_at_start':reset,'placement_offset':actual_placement_offset.tolist(),
                  'gripper_semantics':{'open':0,'closed':1},
                  'physics_backend':'OrcaGymLocal / MuJoCo implicitfast',
                  'camera_alignment':alignment,
                  'camera_metadata':{key:{'name':name,'color_space':'RGB','layout':'HWC','dtype':'uint8',
                                          'renderer':'OrcaLab native','gamma':.45,'resolution':[224,224]}
                                     for key,name in camera_names.items()},
                  'frames':len(rows),'fps':round(1/(robot.dt*robot.frame_skip)),
                  'action_limits':{'joint_rate_rad_s':1.2,'gripper_rate_s':1.6},
                  'simulation_timestep':robot.dt,'frame_skip':robot.frame_skip,
                  'software_versions':{name:importlib.metadata.version(name)
                                       for name in ('orca-lab','orca-gym','mujoco')},
                  'experiment_layout':str(layout.resolve()),
                  'experiment_layout_sha256':layout_hash,
                  'seed':seed,'randomized_reset':randomize,
                  'randomization_profile':randomization_profile,
                  'randomization_config':randomization_config,
                  'approach_jitter_m':approach_jitter}
        (output/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episodes',type=int,default=1)
    parser.add_argument('--target',choices=TARGETS,default=DEFAULT_TARGET)
    parser.add_argument('--profile',choices=EXPERT_PROFILES,default='original')
    parser.add_argument('--layout',type=Path,help='缩小药瓶配置必须显式指定实验场景')
    parser.add_argument('--seed',type=int,default=20261013)
    parser.add_argument('--randomize',action='store_true')
    parser.add_argument('--randomization-profile',choices=PROFILES,default='legacy')
    parser.add_argument('--approach-jitter',type=float,default=0.,
                        help='接近航点每轴随机偏移（米，0–0.03），下探时修正到药盒中心')
    parser.add_argument('--camera-backend',choices=['viewport','sensor'],default='viewport')
    parser.add_argument('--control-hz',type=int,choices=[5,10,20],default=5)
    parser.add_argument('--output',type=Path,default=ROOT/'data/medication_sorting/raw'/datetime.now().strftime('%Y%m%d-%H%M%S'))
    args=parser.parse_args()
    args.layout=resolve_experiment_layout(args.layout, profile=args.profile)
    if not np.isfinite(args.approach_jitter) or not 0<=args.approach_jitter<=.03:
        parser.error('--approach-jitter 必须在0–0.03m')
    args.output.mkdir(parents=True,exist_ok=False)
    robot=FixedBaseRobot(frame_skip=round(1000/args.control_hz))
    results=[]
    try:
        rgb=RenderedRGB(robot) if args.camera_backend=='viewport' else NativeRGB(robot)
        for index in range(args.episodes):
            try:
                report=collect(robot,rgb,args.output/f'episode_{index:04d}',seed=args.seed+index,
                               randomize=args.randomize,approach_jitter=args.approach_jitter,
                               target=args.target,layout=args.layout,profile=args.profile,
                               randomization_profile=args.randomization_profile)
                results.append({'episode':index,'success':report['success']})
            except Exception as exc:
                results.append({'episode':index,'success':False,'error':str(exc)})
                print('回合失败',index,repr(exc),flush=True)
            (args.output/'collection.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
    finally:
        if 'rgb' in locals() and hasattr(rgb,'close'):rgb.close()
        robot.disconnect()
    print(json.dumps(results,ensure_ascii=False,indent=2))
    if not any(row['success'] for row in results):
        raise SystemExit(1)


if __name__=='__main__':main()
