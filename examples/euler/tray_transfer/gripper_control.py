"""Official G1 right-gripper joint2 drive mapping, without model changes."""

from collections.abc import Sequence
from numbers import Real

import numpy as np

OPEN = -1.0
REFERENCE_CLOSE = 2.0

RIGHT_GRIPPER_JOINTS = tuple("unitree_g1_1_" + suffix for suffix in (
    "idx71_gripper_r_inner_joint1",
    "idx72_gripper_r_inner_joint3",
    "idx73_gripper_r_inner_joint4",
    "idx79_gripper_r_inner_joint2",
    "idx81_gripper_r_outer_joint1",
    "idx82_gripper_r_outer_joint3",
    "idx83_gripper_r_outer_joint4",
    "idx89_gripper_r_outer_joint2",
))

JOINT2_DRIVERS = frozenset((
    "unitree_g1_1_idx79_gripper_r_inner_joint2",
    "unitree_g1_1_idx89_gripper_r_outer_joint2",
))


def right_gripper_command(joint_names: Sequence[str], drive: float) -> np.ndarray:
    """Return eight actuator inputs in the supplied joint-name order.

    Only the two joint2 servos receive ``drive``; the other six receive zero.
    They retain their model-defined servos and are not passive joints. ``drive``
    is an actuator input, not a configuration to assign to joint positions.
    Reference open/close values are -1/+2; contact force and grasp success still
    require physics validation for the target object.

    Convention: openverse-orca/SouthGrid@7e98424, g1_pick_osc_conf.py:90-97
    and g1_pick_osc_collection_scripted_lerobot.py:89-91,810-822.
    """
    try:
        names = tuple(joint_names)
    except TypeError as exc:
        raise ValueError("Expected all eight right-gripper joint names") from exc
    if (
        len(names) != 8
        or any(not isinstance(name, str) for name in names)
        or set(names) != set(RIGHT_GRIPPER_JOINTS)
    ):
        raise ValueError("Expected each calibrated unitree_g1_1 right-gripper joint exactly once")
    if (
        isinstance(drive, (bool, np.bool_))
        or not isinstance(drive, Real)
        or not np.isfinite(drive)
        or not OPEN <= drive <= REFERENCE_CLOSE
    ):
        raise ValueError("Gripper drive must be a finite scalar in [-1, 2]")
    return np.array([float(drive) if name in JOINT2_DRIVERS else 0.0 for name in names])


class RightGripperHold:
    """Latch actuator commands during carrying, until an explicit release.

    This holds the command, not the physical joints or the object. Contacts,
    joint motion and slip must still be checked in the simulation.
    """

    def __init__(self, actuator_joint_names: Sequence[str]):
        names = tuple(actuator_joint_names)
        if any(names.count(name) != 1 for name in RIGHT_GRIPPER_JOINTS):
            raise ValueError("Expected each right-gripper actuator exactly once")
        self.command_size = len(names)
        self.indices = np.array([names.index(name) for name in RIGHT_GRIPPER_JOINTS])
        self.held_command: np.ndarray | None = None

    @property
    def active(self) -> bool:
        return self.held_command is not None

    def latch(self, drive: float) -> None:
        command = right_gripper_command(RIGHT_GRIPPER_JOINTS, drive)
        self.latch_command(command)

    def latch_command(self, targets: Sequence[float]) -> None:
        """Hold all eight servo targets, in RIGHT_GRIPPER_JOINTS order.

        Calibrated finger angles must survive arm updates as well as the two
        closing drives. These are actuator commands, never imposed joint poses.
        The caller must validate the model's actuator limits.
        """
        command = np.array(targets, dtype=float, copy=True)
        if command.shape != (8,) or not np.isfinite(command).all():
            raise ValueError("Expected eight finite right-gripper servo targets")
        if np.any(command[[3, 7]] <= 0):
            raise ValueError("Hold requires positive closing drives")
        if self.active and not np.array_equal(command, self.held_command):
            raise ValueError("Release the hold explicitly before changing its targets")
        self.held_command = command

    def release(self) -> None:
        """Allow future commands to open the gripper; does not itself move it."""
        self.held_command = None

    def apply(self, requested: np.ndarray) -> np.ndarray:
        command = np.array(requested, dtype=float, copy=True)
        if command.shape != (self.command_size,) or not np.isfinite(command).all():
            raise ValueError("Expected a finite vector in actuator order")
        if self.active:
            command[self.indices] = self.held_command
        return command
