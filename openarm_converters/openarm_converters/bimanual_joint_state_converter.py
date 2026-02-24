#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory, GripperCommand
from trajectory_msgs.msg import JointTrajectoryPoint
from rclpy.duration import Duration

class BimanualJointStateConverter(Node):
    def __init__(self):
        super().__init__('bimanual_joint_state_converter')

        self.declare_parameter('joint_state_topic', '/joint_states')

        # ★ 전송 주기/시간 파라미터 권장
        self.declare_parameter('command_rate_hz', 20.0)     # 20Hz로만 전송
        self.declare_parameter('execution_time', 0.5)       # 0.3~0.6 권장
        self.declare_parameter('arm_deadband', 0.02)        # rad (≈0.57도)
        self.declare_parameter('gripper_deadband', 0.003)   # 단위에 맞게 조정

        topic = self.get_parameter('joint_state_topic').value

        self.left_arm_client  = ActionClient(self, FollowJointTrajectory, '/left_joint_trajectory_controller/follow_joint_trajectory')
        self.right_arm_client = ActionClient(self, FollowJointTrajectory, '/right_joint_trajectory_controller/follow_joint_trajectory')
        self.left_gripper_client  = ActionClient(self, GripperCommand, '/left_gripper_controller/gripper_cmd')
        self.right_gripper_client = ActionClient(self, GripperCommand, '/right_gripper_controller/gripper_cmd')

        self.subscription = self.create_subscription(JointState, topic, self.joint_state_callback, 10)

        # 최신 목표 저장(콜백에서는 저장만!)
        self.latest_arm_positions = {'left': None, 'right': None}
        self.latest_gripper_positions = {'left': None, 'right': None}

        # 마지막으로 "보낸" 목표 저장
        self.last_sent_arm_positions = {'left': None, 'right': None}
        self.last_sent_gripper_positions = {'left': None, 'right': None}

        # (선택) goal handle 저장해서 preempt 줄이기
        self.arm_goal_handle = {'left': None, 'right': None}
        self.gripper_goal_handle = {'left': None, 'right': None}

        rate = float(self.get_parameter('command_rate_hz').value)
        self.timer = self.create_timer(1.0 / rate, self.control_loop)

        self.get_logger().info(f'Started. Subscribing to {topic}, sending at {rate} Hz')

    def joint_state_callback(self, msg: JointState):
        joint_data = dict(zip(msg.name, msg.position))

        left_arm_joints = [f'openarm_left_joint{i}' for i in range(1, 8)]
        if all(j in joint_data for j in left_arm_joints):
            self.latest_arm_positions['left'] = [joint_data[j] for j in left_arm_joints]

        right_arm_joints = [f'openarm_right_joint{i}' for i in range(1, 8)]
        if all(j in joint_data for j in right_arm_joints):
            self.latest_arm_positions['right'] = [joint_data[j] for j in right_arm_joints]

        for side in ['left', 'right']:
            gripper_joint = None
            for name in [f'openarm_{side}_finger_joint1', f'openarm_{side}_hand']:
                if name in joint_data:
                    gripper_joint = name
                    break
            if gripper_joint:
                self.latest_gripper_positions[side] = float(joint_data[gripper_joint])

    def control_loop(self):
        arm_eps = float(self.get_parameter('arm_deadband').value)
        grip_eps = float(self.get_parameter('gripper_deadband').value)

        # 왼/오른 팔
        for side, client in [('left', self.left_arm_client), ('right', self.right_arm_client)]:
            target = self.latest_arm_positions[side]
            if target is None:
                continue

            last = self.last_sent_arm_positions[side]
            if (last is None) or any(abs(t - l) > arm_eps for t, l in zip(target, last)):
                joint_names = [f'openarm_{side}_joint{i}' for i in range(1, 8)]
                self.send_arm_goal(client, side, joint_names, target)
                self.last_sent_arm_positions[side] = target

        # 왼/오른 그리퍼
        for side, client in [('left', self.left_gripper_client), ('right', self.right_gripper_client)]:
            target = self.latest_gripper_positions[side]
            if target is None:
                continue

            last = self.last_sent_gripper_positions[side]
            if (last is None) or abs(target - last) > grip_eps:
                self.send_gripper_goal(client, side, target)
                self.last_sent_gripper_positions[side] = target

    def send_arm_goal(self, client, side, joint_names, positions):
        if not client.wait_for_server(timeout_sec=0.01):
            return

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = joint_names

        point = JointTrajectoryPoint()
        point.positions = positions

        duration = float(self.get_parameter('execution_time').value)
        point.time_from_start = Duration(seconds=duration).to_msg()

        goal_msg.trajectory.points = [point]

        # (선택) 이전 goal cancel해서 덜덜 줄이기
        prev = self.arm_goal_handle.get(side)
        if prev is not None:
            try:
                prev.cancel_goal_async()
            except Exception:
                pass

        future = client.send_goal_async(goal_msg)
        future.add_done_callback(lambda f: self._store_arm_goal_handle(side, f))

    def _store_arm_goal_handle(self, side, future):
        try:
            self.arm_goal_handle[side] = future.result()
        except Exception:
            self.arm_goal_handle[side] = None

    def send_gripper_goal(self, client, side, position):
        if not client.wait_for_server(timeout_sec=0.01):
            return

        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = position
        goal_msg.command.max_effort = 1.0

        prev = self.gripper_goal_handle.get(side)
        if prev is not None:
            try:
                prev.cancel_goal_async()
            except Exception:
                pass

        future = client.send_goal_async(goal_msg)
        future.add_done_callback(lambda f: self._store_gripper_goal_handle(side, f))

    def _store_gripper_goal_handle(self, side, future):
        try:
            self.gripper_goal_handle[side] = future.result()
        except Exception:
            self.gripper_goal_handle[side] = None

def main(args=None):
    rclpy.init(args=args)
    node = BimanualJointStateConverter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()