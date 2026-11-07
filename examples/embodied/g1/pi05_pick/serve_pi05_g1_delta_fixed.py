"""π0.5 G1 transforms with one seven-joint delta conversion."""
import dataclasses

import numpy as np
from openpi import transforms
from openpi.training import config


@dataclasses.dataclass(frozen=True)
class G1Outputs:
    def __call__(self, data):
        return {"actions": np.asarray(data["actions"])[..., :9]}


@dataclasses.dataclass(frozen=True)
class G1DataConfigFixed(config.LeRobotLiberoDataConfig):
    def create(self, assets_dirs, model_config):
        # Disable the parent class's legacy six-joint conversion. This config
        # adds exactly one conversion for seven arm joints and two absolute
        # finger masters.
        parent = dataclasses.replace(self, extra_delta_transform=False)
        result = config.LeRobotLiberoDataConfig.create(parent, assets_dirs, model_config)
        result = dataclasses.replace(
            result,
            data_transforms=transforms.Group(
                inputs=result.data_transforms.inputs,
                outputs=[G1Outputs()],
            ),
        )
        if self.extra_delta_transform:
            arm_delta_mask = transforms.make_bool_mask(7, -2)
            result = dataclasses.replace(
                result,
                data_transforms=result.data_transforms.push(
                    inputs=[transforms.DeltaActions(arm_delta_mask)],
                    outputs=[transforms.AbsoluteActions(arm_delta_mask)],
                ),
            )
        return result
