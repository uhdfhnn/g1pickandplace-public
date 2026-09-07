"""Dependency-light Cartesian state for the visible G1 keyboard demo.

The Omniverse keyboard subscription and Pinocchio solves live in the simulator
runner. Keeping target updates here makes frame conventions, workspace gating,
gripper commands, and IK accept/reject behavior testable without Isaac Sim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .geometry import Pose


# One key press translates the selected wrist target by 0.01 m (1 cm) along a
# robot-base axis: +X is forward, +Y is left, and +Z is up. This fixed increment
# comes from the smallest centimetre-scale move that remains clearly visible in
# the 640x480 public viewport while avoiding the abrupt contact jumps observed
# with multi-centimetre waypoint changes. The intended operating range is slow,
# discrete tabletop positioning; a larger value can skip through contacts or
# make the next IK target unreachable, while a much smaller value makes manual
# positioning impractically slow. It should become configurable only after a
# visible contact-safety sweep validates alternative step sizes.
CARTESIAN_JOG_STEP_M = 0.01

# Wrist translation is confined to a 0.15 m Euclidean radius around each arm's
# captured reset pose, measured in the robot-base frame with the same +X/+Y/+Z
# convention above. The radius is a provisional local-teleoperation envelope:
# it covers the nearby tabletop manipulation region without authorizing large
# reaches toward the torso, floor, or the opposite side. Too large a radius
# increases self/environment-collision and unreachable-IK risk; too small a
# radius prevents useful scene interaction. Collision-aware IK remains the
# authoritative per-command gate. This value is intentionally fixed pending a
# visible workspace-boundary validation on the public G1/Dex1 asset.
CARTESIAN_WORKSPACE_RADIUS_M = 0.15

# The public Unitree G1/Dex1 action contract exposes exactly seven scalar arm
# joints from shoulder pitch through wrist yaw on each side. This dimension is
# taken from that asset contract, not tuned experimentally. A lower value would
# omit a controllable degree of freedom from IK application; a higher value
# would accept an incompatible action order. It is fixed for this robot asset.
G1_ARM_DOF = 7


@dataclass(frozen=True)
class TeleopEvent:
    """One user-visible result of processing a keyboard press."""

    kind: str
    message: str


class EndEffectorTeleop:
    """Maintain Cartesian wrist requests and limit-clamped joint targets."""

    # Translation deltas use robot-base axes and metres. W/S move along
    # forward/backward X, A/D along left/right Y, and R/F along up/down Z. The
    # mapping follows common planar WASD controls with a vertical pair adjacent
    # to W. Reversing a sign would contradict the printed frame convention and
    # move toward the opposite side of the scene, so the mapping is fixed.
    _CARTESIAN_KEYS = {
        "W": (1.0, 0.0, 0.0),
        "S": (-1.0, 0.0, 0.0),
        "A": (0.0, 1.0, 0.0),
        "D": (0.0, -1.0, 0.0),
        "R": (0.0, 0.0, 1.0),
        "F": (0.0, 0.0, -1.0),
    }

    # These non-Cartesian keys complete the manual-control surface: Q/Escape
    # request shutdown, Tab switches the active arm, and O/C open/close its
    # gripper.  Together with ``_CARTESIAN_KEYS`` they define which physical
    # keyboard events the teleop callback must consume before Isaac Sim's
    # viewport hotkeys see them.  The strings come directly from Carb's key
    # names and have no units or coordinate frame.  Omitting a control key can
    # move the viewport as well as the robot; adding an unrelated key would
    # unnecessarily disable an Isaac Sim shortcut.  This set is intentionally
    # fixed to the printed teleop contract and must change with that contract.
    _COMMAND_KEYS = frozenset(("Q", "ESCAPE", "TAB", "O", "C"))

    @staticmethod
    def _normalized_key(key_name: str) -> str:
        """Return one Carb key name in the controller's canonical form."""

        key = str(key_name).upper()
        return key.removeprefix("KEY_") if key.startswith("KEY_") else key

    @classmethod
    def handles_key(cls, key_name: str) -> bool:
        """Return whether teleop owns this key and must stop its propagation."""

        key = cls._normalized_key(key_name)
        return key in cls._CARTESIAN_KEYS or key in cls._COMMAND_KEYS

    def __init__(
        self,
        *,
        joint_names: Sequence[str],
        initial_positions: Sequence[float],
        joint_limits: Sequence[Sequence[float]],
        initial_poses_by_side: Mapping[str, Pose],
        arm_joints_by_side: Mapping[str, Sequence[str]],
        gripper_joints_by_side: Mapping[str, Sequence[str]],
        gripper_open_positions: Sequence[float],
        gripper_closed_positions: Sequence[float],
        jog_step_m: float = CARTESIAN_JOG_STEP_M,
        workspace_radius_m: float = CARTESIAN_WORKSPACE_RADIUS_M,
    ) -> None:
        self.joint_names = tuple(joint_names)
        self._index = {name: index for index, name in enumerate(self.joint_names)}
        if len(self._index) != len(self.joint_names):
            raise ValueError("joint_names must be unique")

        positions = np.asarray(initial_positions, dtype=np.float64)
        limits = np.asarray(joint_limits, dtype=np.float64)
        if positions.shape != (len(self.joint_names),):
            raise ValueError("initial_positions must match joint_names")
        if limits.shape != (len(self.joint_names), 2):
            raise ValueError("joint_limits must have shape (joint_count, 2)")
        if not np.all(np.isfinite(positions)) or not np.all(np.isfinite(limits)):
            raise ValueError("positions and limits must be finite")
        if np.any(limits[:, 0] > limits[:, 1]):
            raise ValueError("each joint lower limit must be <= its upper limit")
        if not np.isfinite(jog_step_m) or jog_step_m <= 0.0:
            raise ValueError("jog_step_m must be finite and positive")
        if not np.isfinite(workspace_radius_m) or workspace_radius_m <= 0.0:
            raise ValueError("workspace_radius_m must be finite and positive")

        required_sides = ("left", "right")
        self._arm_joints = {
            side: tuple(arm_joints_by_side[side]) for side in required_sides
        }
        self._gripper_joints = {
            side: tuple(gripper_joints_by_side[side]) for side in required_sides
        }
        for side in required_sides:
            if len(self._arm_joints[side]) != G1_ARM_DOF:
                raise ValueError(f"{side} arm must expose exactly {G1_ARM_DOF} joints")
            if len(self._gripper_joints[side]) != 2:
                raise ValueError(f"{side} gripper must expose exactly two joints")
            missing = [
                name
                for name in self._arm_joints[side] + self._gripper_joints[side]
                if name not in self._index
            ]
            if missing:
                raise ValueError(f"{side} teleop joints are absent from action order: {missing}")

        open_positions = np.asarray(gripper_open_positions, dtype=np.float64)
        closed_positions = np.asarray(gripper_closed_positions, dtype=np.float64)
        if open_positions.shape != (2,) or closed_positions.shape != (2,):
            raise ValueError("gripper targets must each contain two positions")
        if not np.all(np.isfinite(open_positions)) or not np.all(np.isfinite(closed_positions)):
            raise ValueError("gripper targets must be finite")

        initial_pose_keys = set(initial_poses_by_side)
        if initial_pose_keys != set(required_sides):
            raise ValueError("initial_poses_by_side must contain exactly left and right")
        reference_poses = {
            side: Pose(
                initial_poses_by_side[side].position,
                initial_poses_by_side[side].quaternion_xyzw,
            )
            for side in required_sides
        }

        self._limits = limits.copy()
        self._targets = np.clip(positions, limits[:, 0], limits[:, 1])
        self._open_positions = open_positions
        self._closed_positions = closed_positions
        self._jog_step_m = float(jog_step_m)
        self._workspace_radius_m = float(workspace_radius_m)
        self._reference_poses = reference_poses
        self._desired_poses = dict(reference_poses)
        self._applied_poses = dict(reference_poses)
        self._requested_revision = {side: 0 for side in required_sides}
        self._applied_revision = {side: 0 for side in required_sides}
        self._next_revision = 0
        self.side = "right"
        self.quit_requested = False

    @property
    def absolute_targets(self) -> np.ndarray:
        """Return a copy so callers cannot mutate controller state."""

        return self._targets.copy()

    def target_pose(self, side: str | None = None) -> Pose:
        """Return the requested wrist pose for one side in robot-base frame."""

        selected_side = self.side if side is None else side
        if selected_side not in self._desired_poses:
            raise ValueError(f"unknown teleop side: {selected_side!r}")
        return self._desired_poses[selected_side]

    @property
    def pending_targets(self) -> tuple[tuple[str, int, Pose], ...]:
        """Return unsolved Cartesian requests in their input order."""

        pending = [
            (side, self._requested_revision[side], self._desired_poses[side])
            for side in ("left", "right")
            if self._requested_revision[side] > self._applied_revision[side]
        ]
        return tuple(sorted(pending, key=lambda item: item[1]))

    def _set_gripper(self, positions: np.ndarray, label: str) -> TeleopEvent:
        for name, position in zip(self._gripper_joints[self.side], positions, strict=True):
            index = self._index[name]
            self._targets[index] = np.clip(
                position,
                self._limits[index, 0],
                self._limits[index, 1],
            )
        return TeleopEvent("gripper", f"{self.side} gripper {label}")

    def accept_ik_solution(
        self,
        side: str,
        revision: int,
        joint_positions: Mapping[str, float],
    ) -> None:
        """Apply one matching IK result to the absolute arm targets."""

        if side not in self._arm_joints:
            raise ValueError(f"unknown teleop side: {side!r}")
        if revision != self._requested_revision[side]:
            raise ValueError("IK solution revision does not match the latest request")
        missing = [name for name in self._arm_joints[side] if name not in joint_positions]
        if missing:
            raise ValueError(f"IK solution is missing {side} arm joints: {missing}")
        for name in self._arm_joints[side]:
            value = float(joint_positions[name])
            if not np.isfinite(value):
                raise ValueError(f"IK solution for {name} is not finite")
            index = self._index[name]
            self._targets[index] = np.clip(
                value,
                self._limits[index, 0],
                self._limits[index, 1],
            )
        self._applied_poses[side] = self._desired_poses[side]
        self._applied_revision[side] = revision

    def reject_ik_solution(self, side: str, revision: int) -> None:
        """Roll an unreachable request back to the last applied wrist pose."""

        if side not in self._arm_joints:
            raise ValueError(f"unknown teleop side: {side!r}")
        if revision != self._requested_revision[side]:
            raise ValueError("IK rejection revision does not match the latest request")
        self._desired_poses[side] = self._applied_poses[side]
        self._applied_revision[side] = revision

    def press(self, key_name: str) -> TeleopEvent | None:
        """Apply one normalized Omniverse key press."""

        key = self._normalized_key(key_name)
        if key in ("Q", "ESCAPE"):
            self.quit_requested = True
            return TeleopEvent("quit", "quit requested")
        if key == "TAB":
            self.side = "left" if self.side == "right" else "right"
            return TeleopEvent("side", f"active arm: {self.side}")
        if key in self._CARTESIAN_KEYS:
            direction = np.asarray(self._CARTESIAN_KEYS[key], dtype=np.float64)
            current_pose = self._desired_poses[self.side]
            candidate_position = current_pose.position + self._jog_step_m * direction
            displacement = candidate_position - self._reference_poses[self.side].position
            if float(np.linalg.norm(displacement)) > self._workspace_radius_m:
                return TeleopEvent(
                    "workspace_limit",
                    f"{self.side} wrist request rejected at "
                    f"{self._workspace_radius_m:.3f} m reset-pose radius",
                )
            self._desired_poses[self.side] = Pose(
                candidate_position,
                current_pose.quaternion_xyzw,
            )
            self._next_revision += 1
            self._requested_revision[self.side] = self._next_revision
            xyz = self._desired_poses[self.side].position
            return TeleopEvent(
                "cartesian",
                f"{self.side} wrist target base xyz=({xyz[0]:+.3f}, {xyz[1]:+.3f}, {xyz[2]:+.3f}) m",
            )
        if key == "O":
            return self._set_gripper(self._open_positions, "open")
        if key == "C":
            return self._set_gripper(self._closed_positions, "closed")
        return None
