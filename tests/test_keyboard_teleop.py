from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from g1pickplace.geometry import Pose
from g1pickplace.keyboard_teleop import EndEffectorTeleop


LEFT_ARM = tuple(f"left_arm_{index}" for index in range(7))
RIGHT_ARM = tuple(f"right_arm_{index}" for index in range(7))
LEFT_GRIPPER = ("left_finger_1", "left_finger_2")
RIGHT_GRIPPER = ("right_finger_1", "right_finger_2")
JOINT_NAMES = LEFT_ARM + RIGHT_ARM + LEFT_GRIPPER + RIGHT_GRIPPER
INITIAL_POSES = {
    "left": Pose(np.asarray([0.3, 0.2, 0.5]), np.asarray([0.0, 0.0, 0.0, 1.0])),
    "right": Pose(np.asarray([0.3, -0.2, 0.5]), np.asarray([0.0, 0.0, 0.0, 1.0])),
}


def _controller(
    *,
    initial: float = 0.0,
    step_m: float = 0.1,
    radius_m: float = 0.25,
) -> EndEffectorTeleop:
    return EndEffectorTeleop(
        joint_names=JOINT_NAMES,
        initial_positions=np.full(len(JOINT_NAMES), initial),
        joint_limits=np.tile((-0.2, 0.2), (len(JOINT_NAMES), 1)),
        initial_poses_by_side=INITIAL_POSES,
        arm_joints_by_side={"left": LEFT_ARM, "right": RIGHT_ARM},
        gripper_joints_by_side={"left": LEFT_GRIPPER, "right": RIGHT_GRIPPER},
        gripper_open_positions=(-0.1, -0.1),
        gripper_closed_positions=(0.15, 0.15),
        jog_step_m=step_m,
        workspace_radius_m=radius_m,
    )


@pytest.mark.parametrize(
    ("key", "delta"),
    (
        ("W", (0.1, 0.0, 0.0)),
        ("S", (-0.1, 0.0, 0.0)),
        ("A", (0.0, 0.1, 0.0)),
        ("D", (0.0, -0.1, 0.0)),
        ("R", (0.0, 0.0, 0.1)),
        ("F", (0.0, 0.0, -0.1)),
    ),
)
def test_right_arm_is_default_and_keys_translate_in_robot_base(
    key: str,
    delta: tuple[float, float, float],
) -> None:
    controller = _controller()
    event = controller.press(key)

    assert event is not None and event.kind == "cartesian"
    np.testing.assert_allclose(
        controller.target_pose("right").position,
        INITIAL_POSES["right"].position + np.asarray(delta),
    )
    np.testing.assert_allclose(
        controller.target_pose("right").quaternion_xyzw,
        INITIAL_POSES["right"].quaternion_xyzw,
    )
    np.testing.assert_allclose(
        controller.target_pose("left").position,
        INITIAL_POSES["left"].position,
    )
    np.testing.assert_allclose(
        controller.target_pose("left").quaternion_xyzw,
        INITIAL_POSES["left"].quaternion_xyzw,
    )


def test_tab_switches_arm_and_multiple_presses_coalesce_into_latest_request() -> None:
    controller = _controller()
    assert controller.press("KEY_TAB").message == "active arm: left"
    controller.press("KEY_W")
    controller.press("A")

    np.testing.assert_allclose(
        controller.target_pose("left").position,
        INITIAL_POSES["left"].position + np.asarray([0.1, 0.1, 0.0]),
    )
    pending = controller.pending_targets
    assert len(pending) == 1
    assert pending[0][0] == "left"
    assert pending[0][1] == 2


def test_reset_pose_workspace_radius_rejects_excess_translation() -> None:
    controller = _controller(step_m=0.1, radius_m=0.15)
    assert controller.press("W").kind == "cartesian"
    event = controller.press("W")

    assert event is not None and event.kind == "workspace_limit"
    np.testing.assert_allclose(
        controller.target_pose("right").position,
        INITIAL_POSES["right"].position + np.asarray([0.1, 0.0, 0.0]),
    )
    assert controller.pending_targets[0][1] == 1


def test_accept_ik_solution_updates_only_selected_arm_and_clears_request() -> None:
    controller = _controller()
    controller.press("W")
    side, revision, _ = controller.pending_targets[0]
    solution = {name: 0.15 for name in RIGHT_ARM}

    controller.accept_ik_solution(side, revision, solution)

    targets = controller.absolute_targets
    np.testing.assert_allclose(
        targets[[JOINT_NAMES.index(name) for name in RIGHT_ARM]],
        0.15,
    )
    np.testing.assert_allclose(
        targets[[JOINT_NAMES.index(name) for name in LEFT_ARM]],
        0.0,
    )
    assert controller.pending_targets == ()


def test_reject_ik_solution_rolls_pose_back_without_joint_change() -> None:
    controller = _controller()
    before = controller.absolute_targets
    controller.press("F")
    side, revision, _ = controller.pending_targets[0]

    controller.reject_ik_solution(side, revision)

    np.testing.assert_allclose(
        controller.target_pose("right").position,
        INITIAL_POSES["right"].position,
    )
    np.testing.assert_allclose(controller.absolute_targets, before)
    assert controller.pending_targets == ()


def test_gripper_commands_apply_only_to_active_side() -> None:
    controller = _controller()
    controller.press("O")
    targets = controller.absolute_targets
    np.testing.assert_allclose(
        targets[[JOINT_NAMES.index(name) for name in RIGHT_GRIPPER]],
        (-0.1, -0.1),
    )
    np.testing.assert_allclose(
        targets[[JOINT_NAMES.index(name) for name in LEFT_GRIPPER]],
        (0.0, 0.0),
    )

    controller.press("TAB")
    controller.press("C")
    targets = controller.absolute_targets
    np.testing.assert_allclose(
        targets[[JOINT_NAMES.index(name) for name in LEFT_GRIPPER]],
        (0.15, 0.15),
    )


def test_quit_keys_are_explicit_and_unknown_keys_are_ignored() -> None:
    controller = _controller()
    assert controller.press("LEFT") is None
    assert not controller.quit_requested
    assert controller.press("KEY_ESCAPE").kind == "quit"
    assert controller.quit_requested


@pytest.mark.parametrize(
    "key",
    ("W", "S", "A", "D", "R", "F", "O", "C", "TAB", "Q", "ESCAPE"),
)
def test_control_keys_are_owned_for_viewport_event_consumption(key: str) -> None:
    assert EndEffectorTeleop.handles_key(key)
    assert EndEffectorTeleop.handles_key(f"KEY_{key}")


def test_unmapped_keys_remain_available_to_isaac_sim() -> None:
    assert not EndEffectorTeleop.handles_key("LEFT")
    assert not EndEffectorTeleop.handles_key("KEY_SPACE")


def test_constructor_rejects_missing_control_joints() -> None:
    with pytest.raises(ValueError, match="absent from action order"):
        EndEffectorTeleop(
            joint_names=JOINT_NAMES[:-1],
            initial_positions=np.zeros(len(JOINT_NAMES) - 1),
            joint_limits=np.tile((-1.0, 1.0), (len(JOINT_NAMES) - 1, 1)),
            initial_poses_by_side=INITIAL_POSES,
            arm_joints_by_side={"left": LEFT_ARM, "right": RIGHT_ARM},
            gripper_joints_by_side={"left": LEFT_GRIPPER, "right": RIGHT_GRIPPER},
            gripper_open_positions=(-0.1, -0.1),
            gripper_closed_positions=(0.1, 0.1),
        )


def test_runner_exits_to_teleop_before_snapshot_planning_or_policy() -> None:
    script = Path(__file__).parents[1] / "scripts" / "run_unitree_mvp.py"
    tree = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    source = ast.get_source_segment(script.read_text(encoding="utf-8"), main)
    assert source is not None
    teleop = source.index("if args.keyboard_teleop:")
    snapshot = source.index("_capture_demo_reset_snapshot(")
    policy = source.index("policy = OpenLoopPolicy(trajectory)")
    assert teleop < snapshot < policy


def test_runner_consumes_all_teleop_owned_keyboard_event_types() -> None:
    script = Path(__file__).parents[1] / "scripts" / "run_unitree_mvp.py"
    source = script.read_text(encoding="utf-8")
    assert "return not controller.handles_key(key_name)" in source
