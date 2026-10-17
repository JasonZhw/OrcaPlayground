"""采集和推理共享的、可复现的直立桌面初态分布。"""

import copy

import numpy as np


PROFILES = {
    'legacy': {'xy_half_range_m': [.006, .008], 'yaw_half_range_rad': .025,
               'upright_only': True},
    'tabletop_v1': {'xy_half_range_m': [.02, .02],
                    'yaw_half_range_rad': float(np.deg2rad(15)), 'upright_only': True},
}


def profile_config(profile='legacy'):
    if profile not in PROFILES:
        raise ValueError(f'未知随机摆放配置：{profile}')
    return copy.deepcopy(PROFILES[profile])


def sample_reset(seed, *, profile='legacy'):
    if type(seed) is not int or seed < 0:
        raise ValueError('随机种子必须是非负整数')
    config = profile_config(profile)
    rng = np.random.default_rng(seed)
    bounds = np.asarray(config['xy_half_range_m'])
    return rng.uniform(-bounds, bounds), float(rng.uniform(-config['yaw_half_range_rad'],
                                                         config['yaw_half_range_rad']))
