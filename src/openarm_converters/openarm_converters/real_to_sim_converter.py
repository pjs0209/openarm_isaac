#!/usr/bin/env python3
# Copyright 2025 Enactic, Inc.
# Licensed under the Apache License, Version 2.0

"""
OpenArm Real-to-Sim Converter (Leader Mode)

- OpenArmCAN으로 상태만 읽어 /joint_states 및 Isaac 피드백 토픽 발행
- 리더 모드에서 토크 OFF 유지 및 LPF 필터링 지원
"""

from typing import List

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

import openarm_can as oa


def _parse_int_list(xs: List[str]) -> List[int]:
    out: List[int] = []
    for x in xs:
        x = str(x).strip()
        if x.lower().startswith("0x"):
            out.append(int(x, 16))
        else:
            out.append(int(x))
    return out


class RealToSimConverter(Node):
    def __init__(self):
        super().__init__("real_to_sim_converter")

        # ---- params ----
        self.declare_parameter("publish_topic", "/joint_states")
        self.declare_parameter("publish_rate_hz", 100.0)

        self.declare_parameter("left_can_interface", "can1")
        self.declare_parameter("right_can_interface", "can0")
        self.declare_parameter("enable_fd", True)

        self.declare_parameter("try_torque_off_on_start", True)
        self.declare_parameter("reassert_torque_off_hz", 1.0)  # 0이면 재전송 안함

        # 추가: LPF 및 피드백 토픽
        self.declare_parameter("enable_lpf", False)
        self.declare_parameter("lpf_alpha", 0.5)
        self.declare_parameter("feedback_publish_topic", "") # 비어있으면 발행 안함

        # candump 기반 기본값
        self.declare_parameter("arm_motor_types", ["DM4310"] * 7)
        self.declare_parameter("arm_send_ids", ["0x01","0x02","0x03","0x04","0x05","0x06","0x07"])
        self.declare_parameter("arm_recv_ids", ["0x11","0x12","0x13","0x14","0x15","0x16","0x17"])

        self.declare_parameter("gripper_motor_type", "DM4310")
        self.declare_parameter("gripper_send_id", "0x08")
        self.declare_parameter("gripper_recv_id", "0x18")

        # URDF와 맞추기
        self.declare_parameter(
            "left_joint_names",
            [
                "openarm_left_joint1","openarm_left_joint2","openarm_left_joint3","openarm_left_joint4",
                "openarm_left_joint5","openarm_left_joint6","openarm_left_joint7","openarm_left_finger_joint1",
            ],
        )
        self.declare_parameter(
            "right_joint_names",
            [
                "openarm_right_joint1","openarm_right_joint2","openarm_right_joint3","openarm_right_joint4",
                "openarm_right_joint5","openarm_right_joint6","openarm_right_joint7","openarm_right_finger_joint1",
            ],
        )
        self.declare_parameter("stamp_now", True)

        # ---- read params ----
        self.pub_topic = self.get_parameter("publish_topic").value
        self.feedback_topic = self.get_parameter("feedback_publish_topic").value
        self.rate_hz = float(self.get_parameter("publish_rate_hz").value)

        self.left_can = self.get_parameter("left_can_interface").value
        self.right_can = self.get_parameter("right_can_interface").value
        self.enable_fd = bool(self.get_parameter("enable_fd").value)

        self.try_torque_off = bool(self.get_parameter("try_torque_off_on_start").value)
        self.reassert_hz = float(self.get_parameter("reassert_torque_off_hz").value)

        self.enable_lpf = bool(self.get_parameter("enable_lpf").value)
        self.lpf_alpha = float(self.get_parameter("lpf_alpha").value)

        arm_types_str = list(self.get_parameter("arm_motor_types").value)
        arm_send_ids = _parse_int_list(list(self.get_parameter("arm_send_ids").value))
        arm_recv_ids = _parse_int_list(list(self.get_parameter("arm_recv_ids").value))

        grip_type_str = str(self.get_parameter("gripper_motor_type").value)
        grip_send_id = _parse_int_list([str(self.get_parameter("gripper_send_id").value)])[0]
        grip_recv_id = _parse_int_list([str(self.get_parameter("gripper_recv_id").value)])[0]

        self.left_joint_names = list(self.get_parameter("left_joint_names").value)
        self.right_joint_names = list(self.get_parameter("right_joint_names").value)
        self.stamp_now = bool(self.get_parameter("stamp_now").value)

        # ---- data structures ----
        self.filtered_pos = {} # name -> value

        # ---- publishers ----
        self.pub = self.create_publisher(JointState, self.pub_topic, 10)
        self.feedback_pub = None
        if self.feedback_topic:
            self.feedback_pub = self.create_publisher(JointState, self.feedback_topic, 10)

        # ---- init OpenArm ----
        self.left_arm = None
        self.right_arm = None
        self._init_can(arm_types_str, arm_send_ids, arm_recv_ids, grip_type_str, grip_send_id, grip_recv_id)

        # timers
        self.timer = self.create_timer(1.0 / self.rate_hz, self._tick)
        self.torque_timer = None
        if self.reassert_hz > 0.0:
            self.torque_timer = self.create_timer(1.0 / self.reassert_hz, self._torque_off_all)

        self.get_logger().info(
            "RealToSimConverter started\n"
            f"  publish_topic: {self.pub_topic}\n"
            f"  feedback_topic: {self.feedback_topic}\n"
            f"  rate_hz      : {self.rate_hz}\n"
            f"  LPF          : {self.enable_lpf} (alpha={self.lpf_alpha})\n"
            f"  left_can     : {self.left_can}\n"
            f"  right_can    : {self.right_can}"
        )

    def _init_can(self, arm_types_str, arm_send_ids, arm_recv_ids, grip_type_str, grip_send_id, grip_recv_id):
        try:
            self.left_arm = oa.OpenArm(self.left_can, self.enable_fd)
            self.right_arm = oa.OpenArm(self.right_can, self.enable_fd)

            def mt(s: str):
                return getattr(oa.MotorType, s)

            arm_types = [mt(s) for s in arm_types_str]
            grip_type = mt(grip_type_str)

            self.left_arm.init_arm_motors(arm_types, arm_send_ids, arm_recv_ids)
            self.right_arm.init_arm_motors(arm_types, arm_send_ids, arm_recv_ids)
            self.left_arm.init_gripper_motor(grip_type, grip_send_id, grip_recv_id)
            self.right_arm.init_gripper_motor(grip_type, grip_send_id, grip_recv_id)

            self.left_arm.set_callback_mode_all(oa.CallbackMode.STATE)
            self.right_arm.set_callback_mode_all(oa.CallbackMode.STATE)

            if self.try_torque_off:
                self._torque_off_all()
            
            self._refresh_and_recv()
        except Exception as e:
            self.get_logger().error(f"Failed to initialize CAN: {e}")

    def _torque_off_all(self):
        try:
            if self.left_arm: self.left_arm.disable_all()
            if self.right_arm: self.right_arm.disable_all()
        except Exception as e:
            self.get_logger().warn(f"disable_all() failed: {e}")

    def _refresh_and_recv(self):
        if self.left_arm:
            self.left_arm.refresh_all()
            self.left_arm.recv_all()
        if self.right_arm:
            self.right_arm.refresh_all()
            self.right_arm.recv_all()

    def _tick(self):
        try:
            self._refresh_and_recv()
        except Exception as e:
            self.get_logger().error(f"CAN refresh/recv failed: {e}")
            return

        msg = JointState()
        if self.stamp_now:
            msg.header.stamp = self.get_clock().now().to_msg()

        names: List[str] = []
        pos: List[float] = []

        # Extract values
        temp_data = [] # (name, raw_pos)
        
        # left
        if self.left_arm:
            lm = self.left_arm.get_arm().get_motors()
            lg = self.left_arm.get_gripper().get_motors()
            for jn, m in zip(self.left_joint_names[:7], lm):
                temp_data.append((jn, float(m.get_position())))
            if lg:
                temp_data.append((self.left_joint_names[7], float(lg[0].get_position())))

        # right
        if self.right_arm:
            rm = self.right_arm.get_arm().get_motors()
            rg = self.right_arm.get_gripper().get_motors()
            for jn, m in zip(self.right_joint_names[:7], rm):
                temp_data.append((jn, float(m.get_position())))
            if rg:
                temp_data.append((self.right_joint_names[7], float(rg[0].get_position())))

        # Apply LPF and build lists
        for name, raw_p in temp_data:
            names.append(name)
            if self.enable_lpf:
                if name in self.filtered_pos:
                    self.filtered_pos[name] = (1.0 - self.lpf_alpha) * self.filtered_pos[name] + self.lpf_alpha * raw_p
                else:
                    self.filtered_pos[name] = raw_p
                pos.append(self.filtered_pos[name])
            else:
                pos.append(raw_p)

        msg.name = names
        msg.position = pos
        
        self.pub.publish(msg)
        if self.feedback_pub:
            self.feedback_pub.publish(msg)


def main():
    rclpy.init()
    node = RealToSimConverter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
