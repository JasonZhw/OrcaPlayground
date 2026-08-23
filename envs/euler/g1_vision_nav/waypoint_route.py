"""Minimal ordered waypoint tracker for the two inspection routes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class RouteUpdate:
    advanced: bool
    completed: bool


class WaypointRoute:
    """Advance through world-frame points using separate intermediate/final tolerances."""

    def __init__(
        self,
        waypoints_xy: Sequence[Sequence[float]],
        *,
        waypoint_tolerance_m: float,
        passage_tolerance_m: float,
        goal_tolerance_m: float,
    ) -> None:
        points = np.asarray(waypoints_xy, dtype=np.float64)
        if points.ndim != 2 or points.shape[0] < 1 or points.shape[1] != 2:
            raise ValueError("waypoints_xy must have shape (N, 2) with N >= 1")
        if not np.all(np.isfinite(points)):
            raise ValueError("waypoints_xy must be finite")
        if min(waypoint_tolerance_m, passage_tolerance_m, goal_tolerance_m) <= 0.0:
            raise ValueError("route tolerances must be positive")
        if passage_tolerance_m < waypoint_tolerance_m:
            raise ValueError("passage_tolerance_m must cover waypoint_tolerance_m")
        self.waypoints = points.copy()
        self.waypoint_tolerance_m = float(waypoint_tolerance_m)
        self.passage_tolerance_m = float(passage_tolerance_m)
        self.goal_tolerance_m = float(goal_tolerance_m)
        self.current_index = 0
        self.completed = False
        self._segment_start: np.ndarray | None = None

    @property
    def current_goal(self) -> np.ndarray:
        return self.waypoints[self.current_index].copy()

    @property
    def final_goal(self) -> np.ndarray:
        return self.waypoints[-1].copy()

    def reset(self) -> None:
        self.current_index = 0
        self.completed = False
        self._segment_start = None

    def update(self, robot_xy: Sequence[float]) -> RouteUpdate:
        position = np.asarray(robot_xy, dtype=np.float64).reshape(2)
        if not np.all(np.isfinite(position)):
            raise ValueError("robot_xy must be finite")
        if self._segment_start is None:
            self._segment_start = position.copy()
        advanced = False
        while not self.completed:
            distance = float(np.linalg.norm(self.current_goal - position))
            final = self.current_index == len(self.waypoints) - 1
            tolerance = self.goal_tolerance_m if final else self.waypoint_tolerance_m
            reached = distance <= tolerance + 1e-6
            passed = not final and self._passed_intermediate_waypoint(position)
            if not reached and not passed:
                break
            if final:
                self.completed = True
                break
            previous_goal = self.current_goal
            self.current_index += 1
            self._segment_start = previous_goal
            advanced = True
        return RouteUpdate(advanced=advanced, completed=self.completed)

    def _passed_intermediate_waypoint(self, position: np.ndarray) -> bool:
        if self._segment_start is None:
            return False
        segment = self.current_goal - self._segment_start
        length_squared = float(np.dot(segment, segment))
        if length_squared <= 1e-12:
            return False
        relative = position - self._segment_start
        progress = float(np.dot(relative, segment) / length_squared)
        if progress < 1.0:
            return False
        cross_product = segment[0] * relative[1] - segment[1] * relative[0]
        cross_track = abs(float(cross_product)) / math.sqrt(length_squared)
        return cross_track <= self.passage_tolerance_m

    def remaining_distance_m(self, robot_xy: Sequence[float]) -> float:
        position = np.asarray(robot_xy, dtype=np.float64).reshape(2)
        if self.completed:
            return float(np.linalg.norm(self.final_goal - position))
        remaining = float(np.linalg.norm(self.current_goal - position))
        for index in range(self.current_index, len(self.waypoints) - 1):
            remaining += float(
                np.linalg.norm(self.waypoints[index + 1] - self.waypoints[index])
            )
        return remaining
