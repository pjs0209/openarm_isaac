#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import time
from collections import deque

class IsaacToOpenArmSmoothDebug(Node):
    """
    1. Sim -> HW (명령 전달): Isaac Sim의 관절 상태를 읽어 하드웨어 컨트롤러로 전달 (LPF/Deadband 적용).
    2. HW -> Sim (상태 전달): 실제 하드웨어의 관절 상태를 읽어 Isaac Sim이 볼 수 있도록 /real_joint_states로 전달.
    """
    def __init__(self):
        super().__init__('isaac_to_openarm_smooth_debug')

        # ===== 1. 명령 전달 필터 (Sim -> HW) 파라미터 =====
        self.declare_parameter('joint_state_topic', '/isaac_joint_states')
        self.declare_parameter('left_cmd_topic',  '/left_forward_position_controller/commands')
        self.declare_parameter('right_cmd_topic', '/right_forward_position_controller/commands')
        self.declare_parameter('left_gripper_cmd_topic',  '/left_gripper_controller/commands')
        self.declare_parameter('right_gripper_cmd_topic', '/right_gripper_controller/commands')
        self.declare_parameter('command_rate_hz', 20.0)
        self.declare_parameter('lpf_alpha', 0.25)
        self.declare_parameter('deadband_ratio', 0.005)
        self.declare_parameter('deadband_min_rad', 0.003)

        # ===== 2. 상태 전달 리레이 (HW -> Sim) 파라미터 =====
        self.declare_parameter('enable_feedback_relay', True)
        self.declare_parameter('hw_joint_state_topic', '/joint_states')
        self.declare_parameter('sim_feedback_topic', '/real_joint_states')

        # 디버그 설정
        self.declare_parameter('debug_print_period', 1.0)
        self.declare_parameter('debug_joint_index', -1)

        # 값 읽기
        self.in_topic = self.get_parameter('joint_state_topic').value
        self.left_cmd_topic  = self.get_parameter('left_cmd_topic').value
        self.right_cmd_topic = self.get_parameter('right_cmd_topic').value
        self.left_gripper_cmd_topic  = self.get_parameter('left_gripper_cmd_topic').value
        self.right_gripper_cmd_topic = self.get_parameter('right_gripper_cmd_topic').value

        self.enable_feedback = self.get_parameter('enable_feedback_relay').value
        self.hw_topic = self.get_parameter('hw_joint_state_topic').value
        self.sim_fw_topic = self.get_parameter('sim_feedback_topic').value

        # ===== 퍼블리셔 생성 =====
        # Sim -> HW 팔/그리퍼 명령들
        self.left_pub  = self.create_publisher(Float64MultiArray, self.left_cmd_topic, 10)
        self.right_pub = self.create_publisher(Float64MultiArray, self.right_cmd_topic, 10)
        self.left_gripper_pub  = self.create_publisher(Float64MultiArray, self.left_gripper_cmd_topic, 10)
        self.right_gripper_pub = self.create_publisher(Float64MultiArray, self.right_gripper_cmd_topic, 10)

        # HW -> Sim 상태 전달용
        if self.enable_feedback:
            self.fw_pub = self.create_publisher(JointState, self.sim_fw_topic, 10)
            self.hw_sub = self.create_subscription(JointState, self.hw_topic, self.hw_cb, 10)

        # ===== 서브스크라이버 생성 (Sim -> HW) =====
        self.sub = self.create_subscription(JointState, self.in_topic, self.cb, 10)

        # 관절 이름 정의
        self.left_arm_joints  = [f'openarm_left_joint{i}' for i in range(1, 8)]
        self.right_arm_joints = [f'openarm_right_joint{i}' for i in range(1, 8)]
        self.left_gripper_joints = ['openarm_left_finger_joint1']
        self.right_gripper_joints = ['openarm_right_finger_joint1']
        
        self.required = set(self.left_arm_joints + self.right_arm_joints + 
                            self.left_gripper_joints + self.right_gripper_joints)

        # 상태 저장 공간
        self.target = {'left': None, 'right': None, 'left_gripper': None, 'right_gripper': None}
        self.cmd = {'left': None, 'right': None, 'left_gripper': None, 'right_gripper': None}

        # 디버그 정보
        self.last_info = {
            'left':  {'updated': False, 'err_max': 0.0},
            'right': {'updated': False, 'err_max': 0.0},
            'left_gripper':  {'updated': False, 'err_max': 0.0},
            'right_gripper': {'updated': False, 'err_max': 0.0},
        }

        self.in_times = deque(maxlen=400)
        self.out_times = deque(maxlen=400)
        self.accepted = 0
        self.dropped = 0
        self.last_drop_reason = ''

        # 타이머
        rate = float(self.get_parameter('command_rate_hz').value)
        self.timer = self.create_timer(1.0 / rate, self.loop)
        self.dbg_timer = self.create_timer(float(self.get_parameter('debug_print_period').value), self.debug_print)

        self.get_logger().info(f"Bridge started. Feedback Relay: {'Enabled' if self.enable_feedback else 'Disabled'}")

    def _hz(self, dq):
        if len(dq) < 2: return 0.0
        dt = dq[-1] - dq[0]
        return (len(dq) - 1) / dt if dt > 0 else 0.0

    def cb(self, msg: JointState):
        """ Isaac Sim -> HW 명령 처리 """
        self.in_times.append(time.time())
        if len(msg.name) != len(msg.position):
            self.dropped += 1
            return
        
        name_set = set(msg.name)
        if not self.required.issubset(name_set):
            self.dropped += 1
            self.last_drop_reason = f'missing joints'
            return

        data = dict(zip(msg.name, msg.position))
        self.target['left']  = [float(data[j]) for j in self.left_arm_joints]
        self.target['right'] = [float(data[j]) for j in self.right_arm_joints]
        self.target['left_gripper']  = [float(data[j]) for j in self.left_gripper_joints]
        self.target['right_gripper'] = [float(data[j]) for j in self.right_gripper_joints]
        self.accepted += 1

    def hw_cb(self, msg: JointState):
        """ HW -> Sim 상태 리레이 처리 """
        # 단순히 토픽명만 변경해서 /real_joint_states로 다시 발행합니다.
        # Isaac Sim은 /real_joint_states 토픽을 구독하여 가상 로봇 모델을 동기화합니다.
        self.fw_pub.publish(msg)

    def loop(self):
        alpha = float(self.get_parameter('lpf_alpha').value)
        dead_ratio = float(self.get_parameter('deadband_ratio').value)
        dead_min = float(self.get_parameter('deadband_min_rad').value)

        self.process('left',  self.left_pub, alpha, dead_ratio, dead_min)
        self.process('right', self.right_pub, alpha, dead_ratio, dead_min)
        self.process('left_gripper',  self.left_gripper_pub, alpha, dead_ratio, dead_min)
        self.process('right_gripper', self.right_gripper_pub, alpha, dead_ratio, dead_min)
        
        self.out_times.append(time.time())

    def process(self, side, pub, alpha, dead_ratio, dead_min):
        tgt = self.target[side]
        if tgt is None: return

        if self.cmd[side] is None:
            self.cmd[side] = tgt[:]
            self.last_info[side]['updated'] = True
        else:
            cur = self.cmd[side]
            n = len(tgt)
            filtered = [(1.0 - alpha) * cur[i] + alpha * tgt[i] for i in range(n)]

            updated = False
            for i in range(n):
                thr = max(abs(cur[i]) * dead_ratio, dead_min)
                if abs(filtered[i] - cur[i]) > thr:
                    updated = True
                    break

            if updated:
                self.cmd[side] = filtered
            self.last_info[side]['updated'] = updated

        err_max = max(abs(tgt[i] - self.cmd[side][i]) for i in range(len(tgt)))
        self.last_info[side]['err_max'] = err_max

        out = Float64MultiArray()
        out.data = self.cmd[side]
        pub.publish(out)

    def debug_print(self):
        in_hz = self._hz(self.in_times)
        out_hz = self._hz(self.out_times)
        self.get_logger().info(
            f"[DBG] IN={in_hz:.1f} OUT={out_hz:.1f} | "
            f"L_err={self.last_info['left']['err_max']:.4f} R_err={self.last_info['right']['err_max']:.4f} | "
            f"LG_err={self.last_info['left_gripper']['err_max']:.4f} RG_err={self.last_info['right_gripper']['err_max']:.4f}"
        )

def main():
    rclpy.init()
    node = IsaacToOpenArmSmoothDebug()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()