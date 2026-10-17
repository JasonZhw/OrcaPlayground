import numpy as np
import pytest
import asyncio
from pathlib import Path
from types import SimpleNamespace
from PIL import Image

from tools.medication_sorting import rendered_rgb


def test_uniform_black_render_cannot_enter_dataset():
    with pytest.raises(ValueError,match='图像'):
        rendered_rgb.validate_rendered_image(np.zeros((224,224,3),dtype=np.uint8))


def test_render_view_uses_o3de_forward_y_camera_convention():
    rotation=np.array([[1,0,0],[0,0,1],[0,-1,0]])
    euler=rendered_rgb.view_euler(rotation)
    np.testing.assert_allclose(euler,[-90,0,0],atol=1e-10)


def test_transient_blank_is_recaptured_at_same_pose_without_stepping(tmp_path):
    calls=[]
    class Client:
        async def call_tool_mcp(self,*,name,arguments,timeout):
            calls.append(arguments)
            pixels=np.zeros((224,224,3),dtype=np.uint8)
            if len(calls)>1:pixels[:,100:]=100
            Image.fromarray(pixels).save(tmp_path/'native.png')
            return SimpleNamespace(model_dump=lambda **kw:{'isError':False})
    camera=object.__new__(rendered_rgb.RenderedRGB)
    camera.client=Client()
    folder=tmp_path/'frame';folder.mkdir()
    rgb,attempts=asyncio.run(camera._capture_slot(folder,'wrist_rgb',
                        np.array([.4,-3.5,5.3]),np.array([-65.,0.,180.]),
                        native_path=tmp_path/'native.png',native_index=301))
    assert len(calls)==2 and calls[0]==calls[1]
    assert attempts==2 and rgb.std()>2
    assert (folder/'rejected_wrist_rgb_01.png').is_file()
    assert (folder/'wrist_rgb.png').is_file()


def test_persistently_blank_camera_is_rejected(tmp_path):
    class Client:
        async def call_tool_mcp(self,**kwargs):
            Image.fromarray(np.zeros((224,224,3),dtype=np.uint8)).save(tmp_path/'native.png')
            return SimpleNamespace(model_dump=lambda **kw:{'isError':False})
    camera=object.__new__(rendered_rgb.RenderedRGB);camera.client=Client()
    with pytest.raises(ValueError,match='图像'):
        asyncio.run(camera._capture_slot(tmp_path,'wrist_rgb',np.zeros(3),np.zeros(3),
                        native_path=tmp_path/'native.png',native_index=301))
