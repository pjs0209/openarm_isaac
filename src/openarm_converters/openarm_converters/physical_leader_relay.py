#!/usr/bin/env python3
# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Physical Leader → Isaac Sim Follower Relay

물리 팔(Leader)의 /joint_states를 읽어
Isaac Sim이 구독하는 /real_joint_states로 relay합니다.

Isaac Sim은 /real_joint_states 토픽을 구독하여
가상 로봇을 물리 팔의 관절 각도로 동기화합니다.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from collections import deque
import time


class PhysicalLeaderToIsaacFollower(Node):
    """
    물리 팔 /joint_states → Isaac Sim /real_joint_states relay 노드.

    - LPF(저역통과필터) 옵션 지원 (Isaac Sim에서 부드럽게 추종)
    - 일정 주기(publish_rate_hz)로 Isaac Sim에 publish
    """

    def __init__(self):
        super().__init__('physical_leader_to_isaac_follower')

        # 파라미터 선언
        self.declare_parameter('hw_joint_state_topic', '/joint_states')
        self.declare_parameter('isaac_feedback_topic', '/real_joint_states')
        self.declare_parameter('publish_rate_hz', 60.0)
        self.declare_parameter('enable_lpf', False)
        self.declare_parameter('lpf_alpha', 0.5)   # 0=변화없음, 1=필터없음
        self.declare_parameter('debug_print_period', 2.0)

        # 파라미터 읽기
        hw_topic = self.get_parameter('hw_joint_state_topic').value
        isaac_topic = self.get_parameter('isaac_feedback_topic').value
        rate_hz = float(self.get_parameter('publish_rate_hz').value)
        self.enable_lpf = self.get_parameter('enable_lpf').value
        self.lpf_alpha = float(self.get_parameter('lpf_alpha').value)

        # 상태 저장
        self.latest_msg: JointState | None = None
        self.filtered_positions: dict[str, float] = {}

        # 디버그용
        self.in_times = deque(maxlen=200)
        self.out_times = deque(maxlen=200)

        # Publisher: Isaac Sim으로 전달
        self.isaac_pub = self.create_publisher(JointState, isaac_topic, 10)

        # Subscriber: 물리 팔 상태 수신
        self.hw_sub = self.create_subscription(
            JointState, hw_topic, self._hw_cb, 10
        )

        # 주기적 publish 타이머
        self.pub_timer = self.create_timer(1.0 / rate_hz, self._publish_loop)

        # 디버그 타이머
        dbg_period = float(self.get_parameter('debug_print_period').value)
        self.dbg_timer = self.create_timer(dbg_period, self._debug_print)

        self.get_logger().info(
            f'PhysicalLeaderToIsaacFollower started\n'
            f'  HW topic    : {hw_topic}\n'
            f'  Isaac topic : {isaac_topic}\n'
            f'  Rate        : {rate_hz} Hz\n'
            f'  LPF enabled : {self.enable_lpf} (alpha={self.lpf_alpha})'
        )

    def _hw_cb(self, msg: JointState):
        """물리 팔 joint_states 수신 콜백."""
        self.in_times.append(time.time())

        if self.enable_lpf:
            # LPF 적용: 이전 값과 new 값을 가중 평균
            for name, pos in zip(msg.name, msg.position):
                if name in self.filtered_positions:
                    self.filtered_positions[name] = (
                        (1.0 - self.lpf_alpha) * self.filtered_positions[name]
                        + self.lpf_alpha * pos
                    )
                else:
                    self.filtered_positions[name] = pos

            # 필터링된 값으로 메시지 업데이트
            filtered_msg = JointState()
            filtered_msg.header = msg.header
            filtered_msg.name = list(msg.name)
            filtered_msg.position = [
                self.filtered_positions.get(n, p)
                for n, p in zip(msg.name, msg.position)
            ]
            filtered_msg.velocity = list(msg.velocity)
            filtered_msg.effort = list(msg.effort)
            self.latest_msg = filtered_msg
        else:
            # 필터 없이 그대로 전달
            self.latest_msg = msg

    def _publish_loop(self):
        """Isaac Sim으로 주기적 publish."""
        if self.latest_msg is None:
            return

        self.isaac_pub.publish(self.latest_msg)
        self.out_times.append(time.time())

    def _hz(self, dq: deque) -> float:
        if len(dq) < 2:
            return 0.0
        dt = dq[-1] - dq[0]
        return (len(dq) - 1) / dt if dt > 0 else 0.0

    def _debug_print(self):
        in_hz = self._hz(self.in_times)
        out_hz = self._hz(self.out_times)
        received = self.latest_msg is not None
        self.get_logger().info(
            f'[Status] IN={in_hz:.1f}Hz  OUT={out_hz:.1f}Hz  '
            f'msg_received={received}'
        )


def main():
    rclpy.init()
    node = PhysicalLeaderToIsaacFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
