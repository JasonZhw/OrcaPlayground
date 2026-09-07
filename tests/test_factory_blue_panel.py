"""终点蓝色构图确认：不连接仿真，用真实 RGB 检测和终点控制器。"""

import numpy as np
import pytest

from envs.euler.g1_vision_nav.config import BluePanelInspectionConfig
from envs.euler.g1_vision_nav.factory_inspection_navigator import BluePanelInspection


def panel(x=100, color=(25, 65, 170)):
    image = np.full((200, 300, 3), 120, np.uint8)
    image[45:115, x:x + 90] = color
    return image


def tick(controller, image, index, seconds, **kwargs):
    values = dict(rgb=image, frame_index=index, sim_time_s=seconds,
                  position_xy=(19.2, 4.3), heading_rad=0.,
                  linear_speed_mps=0., yaw_rate_rps=0., command_still=True, safe=True)
    values.update(kwargs)
    return controller.update(**values)


def inspector():
    controller = BluePanelInspection(BluePanelInspectionConfig(stable_s=.5))
    controller.start((19.2, 4.3), 0.)
    tick(controller, None, 0, 0.)  # 到点后的第一拍必须先发站立指令。
    return controller


def test_center_blue_requires_stable_new_frames_and_copies_exact_photo():
    controller = inspector()
    image = panel()
    for i in range(5):
        command = tick(controller, image, i + 1, i * .1)
        assert not controller.ready
        assert command.forward_mps == command.lateral_mps == 0
    tick(controller, image, 6, .5)
    assert controller.ready
    np.testing.assert_array_equal(controller.photo_rgb, image)
    image[:] = 0
    assert controller.photo_rgb.any()


@pytest.mark.parametrize('color', [(120, 120, 120), (180, 40, 25), (25, 170, 65), (2, 3, 8)])
def test_non_blue_or_dark_frames_never_qualify(color):
    controller = inspector()
    for i in range(15):
        tick(controller, panel(color=color), i + 1, i * .1)
    assert not controller.ready


@pytest.mark.parametrize('x', [35, 175])
def test_blue_off_center_stops_and_takes_photo_without_turning(x):
    controller = inspector()
    for i in range(6):
        command = tick(controller, panel(x), i + 1, i * .1)
        assert not command.walk_enabled
        assert command.yaw_rate_rps == 0
    assert controller.ready


def test_arrival_stops_before_inspection_even_when_no_blue():
    controller = BluePanelInspection()
    controller.start((19.2, 4.3), 0.)
    for i in range(3):
        command = tick(controller, panel(color=(120, 120, 120)), i + 1, i * .1,
                       linear_speed_mps=.2, heading_rad=-1.)
        assert not command.walk_enabled
    command = tick(controller, panel(color=(120, 120, 120)), 4, .3)
    assert command.walk_enabled and command.yaw_rate_rps != 0


def test_thin_blue_shelving_and_full_blue_image_do_not_qualify():
    image = panel(color=(120, 120, 120))
    image[20:160, 120:124] = (25, 65, 170)
    for rgb in (image, np.full_like(image, (25, 65, 170))):
        controller = inspector()
        for i in range(12):
            tick(controller, rgb, i + 1, i * .1)
        assert not controller.ready


def test_repeated_frame_and_outage_cannot_complete_confirmation():
    controller = inspector()
    for i in range(20):
        tick(controller, panel(), 1, i * .1)
    assert not controller.ready
    tick(controller, None, 0, 2.)
    tick(controller, panel(), 2, 2.1)
    assert not controller.ready


@pytest.mark.parametrize('extra', [dict(linear_speed_mps=.2), dict(yaw_rate_rps=.3),
                                   dict(command_still=False), dict(safe=False)])
def test_motion_or_collision_resets_confirmation(extra):
    controller = inspector()
    for i in range(5):
        tick(controller, panel(), i + 1, i * .1)
    tick(controller, panel(), 6, .5, **extra)
    tick(controller, panel(), 7, .6)
    assert not controller.ready


def test_search_timeout_stops_without_photo():
    controller = inspector()
    command = tick(controller, panel(), 1, 46.)
    assert controller.failed and not controller.ready
    assert controller.photo_rgb is None
    assert command.forward_mps == command.lateral_mps == command.yaw_rate_rps == 0


def test_not_started_does_not_take_blue_photo_along_route():
    controller = BluePanelInspection()
    for i in range(10):
        tick(controller, panel(), i + 1, i * .1)
    assert not controller.ready


def test_safety_invalidation_discards_frame_but_preserves_total_timeout():
    controller = inspector()
    for i in range(7):
        tick(controller, panel(), i + 1, i * .1)
    assert controller.ready
    controller.invalidate_confirmation()
    assert not controller.ready and controller.photo_rgb is None
    tick(controller, panel(), 20, 45.)
    assert controller.failed and not controller.ready


def test_visible_blue_is_accepted_without_world_heading_gate():
    controller = inspector()
    for i in range(12):
        tick(controller, panel(), i + 1, i * .1, heading_rad=np.pi)
    assert controller.ready


@pytest.mark.parametrize('extra', [dict(heading_rad=float('nan')),
                                  dict(position_xy=(float('inf'), 0.)),
                                  dict(yaw_rate_rps=float('nan'))])
def test_invalid_pose_fails_closed(extra):
    controller = inspector()
    command = tick(controller, panel(), 1, .1, **extra)
    assert controller.failed and not controller.ready
    assert command.yaw_rate_rps == 0


def test_new_frame_gap_and_visual_loss_restart_stable_timer():
    controller = inspector()
    for i in range(4):
        tick(controller, panel(), i + 1, i * .1)
    tick(controller, panel(), 10, 1.)
    assert not controller.ready
    tick(controller, panel(color=(120, 120, 120)), 11, 1.1)
    tick(controller, panel(), 12, 1.2)
    assert not controller.ready


def test_no_blue_searches_one_direction_then_times_out():
    controller = inspector()
    heading = np.arctan2(.06, 2.39)
    for i in range(20):
        command = tick(controller, panel(color=(120, 120, 120)), i + 1, i * .1,
                       heading_rad=heading)
        assert command.yaw_rate_rps > 0
    assert not controller.ready
    command = tick(controller, panel(), 30, 15.)
    assert controller.failed and not command.walk_enabled


def test_losing_blue_resumes_search_but_visible_off_center_blue_stops():
    controller = inspector()
    tick(controller, panel(), 1, 0.)
    for i, image in enumerate((panel(35), panel(color=(120, 120, 120)), panel(175)), 2):
        command = tick(controller, image, i, i * .1)
        assert (command.yaw_rate_rps != 0) == (i == 3)
        assert not controller.ready


def test_drift_stops_search_without_permanently_rejecting_photo():
    controller = inspector()
    command = tick(controller, panel(color=(120, 120, 120)), 1, .1, position_xy=(19.64, 4.3))
    assert not controller.failed and not controller.ready
    assert command.yaw_rate_rps == 0
    for i in range(6):
        tick(controller, panel(), i + 2, .2 + i * .1, position_xy=(19.64, 4.3))
    assert controller.ready


def test_default_captures_fresh_five_percent_panel_after_drift_without_centering():
    controller = BluePanelInspection()
    controller.start((20., 3.8), 0.)
    image = np.full((200, 300, 3), 120, np.uint8)
    image[50:102, 220:260] = (25, 65, 170)  # ROI 的 5.42%，偏右。
    pose = dict(position_xy=(20.22, 3.43), heading_rad=np.deg2rad(47))
    assert not tick(controller, image, 1, 0., **pose).walk_enabled
    command = tick(controller, image, 2, .1, **pose)
    assert .05 <= controller.blue_fraction < .06
    assert controller.ready and not controller.failed
    assert not command.walk_enabled
    np.testing.assert_array_equal(controller.photo_rgb, image)


def test_below_five_percent_is_not_enough_for_default_photo():
    controller = BluePanelInspection()
    controller.start((19.2, 4.3), 0.)
    image = np.full((200, 300, 3), 120, np.uint8)
    image[50:94, 220:260] = (25, 65, 170)  # ROI 的 4.58%。
    for i in range(8):
        tick(controller, image, i + 1, i * .1)
    assert not controller.ready


@pytest.mark.parametrize('extra', [dict(safe=False), dict(linear_speed_mps=.2),
                                  dict(yaw_rate_rps=.3), dict(command_still=False)])
def test_immediate_photo_still_requires_safety_and_standstill(extra):
    controller = BluePanelInspection()
    controller.start((19.2, 4.3), 0.)
    tick(controller, panel(), 1, 0.)
    for i in range(5):
        tick(controller, panel(), i + 2, .1 + i * .1, **extra)
    assert not controller.ready and controller.photo_rgb is None


def test_drift_on_repeated_frame_stops_turning_without_using_old_pixels():
    controller = BluePanelInspection()
    controller.start((19.2, 4.3), 0.)
    tick(controller, None, 0, 0.)
    command = tick(controller, panel(color=(120, 120, 120)), 1, .1)
    assert command.yaw_rate_rps != 0
    command = tick(controller, panel(), 1, .2, position_xy=(19.65, 4.3))
    assert not command.walk_enabled and not controller.ready
    tick(controller, panel(), 2, .3, position_xy=(19.65, 4.3))
    assert controller.ready


def test_inspection_accepts_new_arrival_radius_and_uses_cabinet_log():
    controller = inspector()
    for i in range(6):
        tick(controller, panel(), i + 1, i * .1, position_xy=(19.58, 4.3))
        if i < 5:
            assert controller.mode == "检测到电气柜，保持站立确认照片"
    assert controller.ready and not controller.failed
