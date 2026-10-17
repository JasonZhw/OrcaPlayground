"""OrcaLab 原生视口渲染的固定俯视和虚拟腕部相机。

传入的是相机位姿，不是机器人/药盒目标位姿；仿真暂停时截图，
训练与推理使用同一视角。独立 PNG 和相机变换保留供同步复查。
"""

import asyncio
from functools import partial
import json
import os
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from tools.medication_sorting.robot_control import PREFIX

BASE_POSITION = np.array([.58,-3.62,5.35])
BASE_ROTATION = np.array([-90.,0.,180.])


def view_euler(rotation):
    return Rotation.from_matrix(rotation).as_euler('xyz',degrees=True)


def validate_rendered_image(rgb):
    if rgb.dtype != np.uint8 or rgb.shape != (224,224,3):
        raise ValueError('渲染图像的尺寸或类型不符')
    if rgb.std()<2 or np.mean(rgb>8)<.05:
        raise ValueError('渲染图像空白或有效像素不足')
    return rgb


class RenderedRGB:
    camera_names={'base_rgb':'virtual_overhead','wrist_rgb':'virtual_hand_follower'}
    alignment='paused_physics_snapshot_with_explicit_camera_transform'

    def __init__(self,robot):
        import httpx2
        from fastmcp import Client
        from fastmcp.client.transports import StreamableHttpTransport
        self.robot=robot
        transport=StreamableHttpTransport('http://127.0.0.1:12345/mcp',
                      httpx_client_factory=partial(httpx2.AsyncClient,trust_env=False))
        self.client=Client(transport,timeout=20,init_timeout=10)
        robot.loop.run_until_complete(self.client.__aenter__())

    async def _capture_slot(self,folder,slot,position,rotation,*,native_path,native_index):
        # 个别原生渲染请求会返回全黑 PNG。保持物理状态和相机不变重绘，
        # 保留被拒画面；持续黑图仍使回合失败，不用前一帧填补观测。
        for attempt in range(1,4):
            previous=native_path.stat().st_mtime_ns if native_path.exists() else None
            result=await self.client.call_tool_mcp(name='get_view_png_from_transform',
                    arguments={'position':np.asarray(position).tolist(),
                               'rotation':np.asarray(rotation).tolist(),'index':native_index},timeout=20)
            if result.model_dump(by_alias=True).get('isError'):
                raise RuntimeError('OrcaLab 相机渲染请求失败')
            if not native_path.exists() or native_path.stat().st_mtime_ns==previous:
                raise RuntimeError('OrcaLab 未生成新的相机画面')
            with Image.open(native_path) as im:
                pixels=np.asarray(im.convert('RGB').resize((224,224),Image.Resampling.LANCZOS))
            pixels=np.rint(255*(pixels.astype(np.float32)/255)**.45).astype(np.uint8)
            try:
                validate_rendered_image(pixels)
            except ValueError:
                shutil.copyfile(native_path,folder/f'rejected_{slot}_{attempt:02d}.png')
                if attempt==3:raise
                print(f'相机 {slot} 空白，保持当前状态重绘 {attempt}/3',flush=True)
                continue
            shutil.copyfile(native_path,folder/(slot+'.png'))
            Image.fromarray(pixels).save(folder/(slot+'_224.png'))
            return pixels,attempt

    async def _capture(self,folder,index):
        # 末端相机的坐标来自实际腕部刚体，不跟踪药盒真值。
        # 原腕部传感器的外伸安装位置会越过房间边界；虚拟观察相机跟随
        # 实际末端位置，固定世界系偏移与朝向。此视角配置在推理时完全复用。
        ee=self.robot.site_pose()['xpos']
        wrist_position=ee+np.array([-.035,.16,.28])
        poses={'base_rgb':(BASE_POSITION,BASE_ROTATION),
               'wrist_rgb':(wrist_position,np.array([-65.,0.,180.]))}
        images={};attempts={}
        for slot,(position,rotation) in poses.items():
            native_index=300 if slot=='base_rgb' else 301
            native_directory=Path(os.environ.get('ORCALAB_CAPTURE_DIR', Path.home()/'.orcalab')).expanduser()
            native_path=native_directory/f'view_transform_{native_index}.png'
            images[slot],attempts[slot]=await self._capture_slot(folder,slot,position,rotation,
                                                    native_path=native_path,native_index=native_index)
        (folder/'camera_transforms.json').write_text(json.dumps({
            'frame':index,'sim_time':self.robot.sim_time,'physics_paused':True,
            'ee_position':ee.tolist(),'render_attempts':attempts,
            'poses':{k:{'position':p.tolist(),'rotation_degrees':r.tolist()}
                     for k,(p,r) in poses.items()}},indent=2))
        return images

    def capture(self,folder,index):
        folder.mkdir(parents=True,exist_ok=False)
        self.robot.render(index)
        return self.robot.loop.run_until_complete(self._capture(folder,index))

    def close(self):
        self.robot.loop.run_until_complete(self.client.__aexit__(None,None,None))
