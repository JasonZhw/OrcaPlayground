"""四物品任务的唯一标签及专家试验参数；可供离线验收读取，不加载仿真。"""

DEFAULT_TARGET = 'pharyngitis_tablets_box_1'
TARGETS = {
    'calcium_gluconate_tablets_bottle_1': {
        'body_suffix':'_Medical_32', 'basket_suffix':'_Box_Blue',
        'task':'Put the calcium gluconate tablets bottle into the blue basket.',
        'travel_z':5.12, 'initial_closure':0., 'grasp_clearance':.06, 'lower_z':5.04,
    },
    'melatonin_tablet_bottle_1': {
        'body_suffix':'_Medical_31', 'basket_suffix':'_Box_Blue',
        'task':'Put the melatonin tablet bottle into the blue basket.',
        'travel_z':5.12, 'initial_closure':0., 'grasp_clearance':.05, 'lower_z':5.04,
        'tilt_x_degrees':-10.,
    },
    'fu_shi_gao_herb_drink_packaging_box_1': {
        'body_suffix':'_Medical_23', 'basket_suffix':'_Box_Purple',
        'task':'Put the Fu Shi Gao herb drink box into the purple basket.',
        'travel_z':5.10, 'initial_closure':.5,
    },
    DEFAULT_TARGET: {
        'body_suffix':'_Medical_12', 'basket_suffix':'_Box_Purple',
        'task':'Put the medicine box into the purple basket.',
        'travel_z':5.10, 'initial_closure':.5,
    },
}

EXPERT_PROFILES = ('original', 'scaled_bottles_70', 'randomized_tabletop_70')


def target_spec(target, *, profile='original'):
    if profile not in EXPERT_PROFILES:
        raise ValueError(f'未知专家配置：{profile}')
    if target not in TARGETS:
        raise ValueError(f'不支持的物品：{target}')
    spec = dict(TARGETS[target])
    if profile in ('scaled_bottles_70', 'randomized_tabletop_70') and 'bottle' in target:
        spec.update(travel_z=5.04, initial_closure=0., grasp_clearance=-.005,
                    lower_z=4.99 if 'melatonin' in target else 4.91,
                    tilt_x_degrees=-10. if 'melatonin' in target else 0.)
    if profile == 'randomized_tabletop_70' and target == 'fu_shi_gao_herb_drink_packaging_box_1':
        spec.update(initial_closure=0., placement_bias=(0., .025),
                    expert_revision='herb_release_offset_v1')
    return spec
