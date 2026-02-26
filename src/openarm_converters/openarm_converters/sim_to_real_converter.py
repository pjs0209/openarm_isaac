#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import time
from collections import deque

class SimToRealConverter(Node):
    """
    Isaac Sim -> HW (명령 전달): Isaac Sim의 관절 상태를 읽어 하드웨어 컨트롤러로 전달 (LPF/Deadband 적용).
    """
    def __init__(self):
        super().__init__('sim_to_real_converter')

        # ===== 명령 전달 필터 (Sim -> HW) 파라미터 =====
        self.declare_parameter('joint_state_topic', '/isaac_joint_states')
        self.declare_parameter('left_cmd_topic',  '/left_forward_position_controller/commands')
        self.declare_parameter('right_cmd_topic', '/right_forward_position_controller/commands')
        self.declare_parameter('left_gripper_cmd_topic',  '/left_gripper_controller/commands')
        self.declare_parameter('right_gripper_cmd_topic', '/right_gripper_controller/commands')
        self.declare_parameter('command_rate_hz', 50.0)
        self.declare_parameter('lpf_alpha', 0.25)
        self.declare_parameter('deadband_ratio', 0.005)
        self.declare_parameter('deadband_min_rad', 0.003)
        self.declare_parameter('timeout_seconds', 0.5)

        # 디버그 설정
        self.declare_parameter('debug_print_period', 1.0)

        # 값 읽기
        self.in_topic = self.get_parameter('joint_state_topic').value
        self.left_cmd_topic  = self.get_parameter('left_cmd_topic').value
        self.right_cmd_topic = self.get_parameter('right_cmd_topic').value
        self.left_gripper_cmd_topic  = self.get_parameter('left_gripper_cmd_topic').value
        self.right_gripper_cmd_topic = self.get_parameter('right_gripper_cmd_topic').value
        self.timeout_sec = self.get_parameter('timeout_seconds').value

        # ===== 퍼블리셔 생성 =====
        # Sim -> HW 팔/그리퍼 명령들
        self.left_pub  = self.create_publisher(Float64MultiArray, self.left_cmd_topic, 10)
        self.right_pub = self.create_publisher(Float64MultiArray, self.right_cmd_topic, 10)
        self.left_gripper_pub  = self.create_publisher(Float64MultiArray, self.left_gripper_cmd_topic, 10)
        self.right_gripper_pub = self.create_publisher(Float64MultiArray, self.right_gripper_cmd_topic, 10)

        # ===== 서브스크라이버 생성 (Sim -> HW) =====
        self.sub = self.create_subscription(JointState, self.in_topic, self.cb, 10)

        # 관절 이름 정의
        self.left_arm_joints  = [f'openarm_left_joint{i}' for i in range(1, 8)]
        self.right_arm_joints = [f'openarm_right_joint{i}' for i in range(1, 8)]
        self.left_gripper_joints = ['openarm_left_finger_joint1']
        self.right_gripper_joints = ['openarm_right_finger_joint1']
        
        self.required = self.left_arm_joints + self.right_arm_joints + \
                        self.left_gripper_joints + self.right_gripper_joints

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
        self.last_msg_time = 0.0
        self.joint_mapping = {}

        # 타이머
        rate = float(self.get_parameter('command_rate_hz').value)
        self.timer = self.create_timer(1.0 / rate, self.loop)
        self.dbg_timer = self.create_timer(float(self.get_parameter('debug_print_period').value), self.debug_print)

        self.get_logger().info("SimToRealConverter bridge started.")

    def _hz(self, dq):
        if len(dq) < 2: return 0.0
        dt = dq[-1] - dq[0]
        return (len(dq) - 1) / dt if dt > 0 else 0.0

    def _update_joint_mapping(self, msg_names):
        """ Isaac Sim에서 오는 이름들(msg_names)에서 필요한 이름(self.required)을 찾아 매핑 생성 """
        self.joint_mapping = {}
        for req in self.required:
            # 1. Exact match
            if req in msg_names:
                self.joint_mapping[req] = req
                continue
            # 2. Suffix match (Isaac Sim의 경우 /World/OpenArm/openarm_left_joint1 처럼 올 수 있음)
            found = False
            for m_name in msg_names:
                if m_name.endswith(req):
                    self.joint_mapping[req] = m_name
                    found = True
                    break
            if not found:
                self.get_logger().warn(f"Joint '{req}' not found in message names by suffix match.")

    def cb(self, msg: JointState):
        """ Isaac Sim -> HW 명령 처리 """
        now = time.time()
        self.in_times.append(now)
        self.last_msg_time = now

        if len(msg.name) != len(msg.position):
            self.dropped += 1
            self.last_drop_reason = 'mismatched name/pos length'
            return
        
        # 이름 매핑이 없거나 메시지 구성을 보고 갱신이 필요하면 수행
        if not self.joint_mapping or len(set(msg.name)) != len(self.joint_mapping):
             self._update_joint_mapping(msg.name)

        data = dict(zip(msg.name, msg.position))
        
        def extract(joints):
            vals = []
            for j in joints:
                m_name = self.joint_mapping.get(j)
                if m_name and m_name in data:
                    vals.append(float(data[m_name]))
                else:
                    return None
            return vals

        l_arm = extract(self.left_arm_joints)
        r_arm = extract(self.right_arm_joints)
        l_grip = extract(self.left_gripper_joints)
        r_grip = extract(self.right_gripper_joints)

        if any(v is None for v in [l_arm, r_arm, l_grip, r_grip]):
            self.dropped += 1
            self.last_drop_reason = 'missing joints after mapping'
            return

        self.target['left']  = l_arm
        self.target['right'] = r_arm
        self.target['left_gripper']  = l_grip
        self.target['right_gripper'] = r_grip
        self.accepted += 1

    def loop(self):
        # 타임아웃 체크: Isaac Sim으로부터 데이터가 너무 오랫동안 안 오면 명령 전송 중단
        if self.last_msg_time > 0 and (time.time() - self.last_msg_time) > self.timeout_sec:
            return

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
        status = "OK"
        if self.last_msg_time > 0 and (time.time() - self.last_msg_time) > self.timeout_sec:
            status = "TIMEOUT"

        self.get_logger().info(
            f"[Bridge:{status}] IN={in_hz:.1f} OUT={out_hz:.1f} | "
            f"L_err={self.last_info['left']['err_max']:.4f} R_err={self.last_info['right']['err_max']:.4f} | "
            f"LG_err={self.last_info['left_gripper']['err_max']:.4f} RG_err={self.last_info['right_gripper']['err_max']:.4f}"
        )

def main():
    rclpy.init()
    node = SimToRealConverter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()