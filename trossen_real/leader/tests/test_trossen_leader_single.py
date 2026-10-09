import pytest

from trossen_real.leader.trossen_leader_single import TrossenSingleLeader


class FakeArm:
    def __init__(self):
        self.commands = []

    def set_joint_positions(self, joint_positions, goal_time, blocking):
        self.commands.append(list(joint_positions))

    def set_mode_position(self):
        pass

    def set_mode_freedrive(self):
        pass

    def disconnect(self):
        pass


def test_clips_gripper_for_tracking_sync_and_park():
    leader = TrossenSingleLeader("unused", gripper_bounds=(0.004, 0.04))
    arm = FakeArm()
    leader._arm = arm
    leader._current_mode = "position"

    leader.track_joints([0.0] * 6 + [0.044], goal_time=0.1)
    leader.sync_to_joints([0.0] * 6 + [0.044], goal_time=1.0, end_in_freedrive=False)
    leader.disconnect(park_waypoints=[[0.0] * 6 + [0.044]], goal_time=1.0)

    assert [command[6] for command in arm.commands] == pytest.approx([0.04, 0.04, 0.04])