#!/usr/bin/env python3

# 필수 라이브러리들을 불러옵니다.
import rclpy  # ROS2의 파이썬 클라이언트 라이브러리
from rclpy.node import Node  # ROS2 노드를 만들기 위한 기본 클래스
from rclpy.action import ActionClient  # 액션 서버에 명령을 보내기 위한 클라이언트
from sensor_msgs.msg import JointState  # 로봇의 관절 상태 메시지 형식
from control_msgs.action import FollowJointTrajectory, GripperCommand  # 팔 궤적 및 그리퍼 제어 액션 형식
from trajectory_msgs.msg import JointTrajectoryPoint  # 궤적의 한 지점(위치, 시간 등)을 정의하는 형식

class BimanualJointStateConverter(Node):
    """
    이 클래스는 '토픽(Topic)'으로 들어오는 로봇의 관절 상태를 받아,
    실제 로봇이 움직일 수 있도록 '액션(Action)' 명령으로 변환하여 전송하는 역할을 합니다.
    """
    def __init__(self):
        # 노드 이름을 'bimanual_joint_state_converter'로 초기화합니다.
        super().__init__('bimanual_joint_state_converter')

        # --- 설정 파라미터 등록 ---
        # 외부에서 변경 가능한 변수들을 정의합니다. (기본값 설정)
        self.declare_parameter('joint_state_topic', '/joint_states') # 구독할 관절 상태 토픽 이름
        self.declare_parameter('execution_time', 2)  # 목표 위치까지 도달하는 데 걸리는 시간 (초)

        # 설정된 토픽 이름을 가져옵니다.
        topic = self.get_parameter('joint_state_topic').get_parameter_value().string_value
        
        # --- 액션 클라이언트 준비 ---
        # 실제 로봇 컨트롤러(액션 서버)와 통신하기 위한 클라이언트를 만듭니다.
        # 왼쪽 팔 궤적 제어
        self.left_arm_client = ActionClient(self, FollowJointTrajectory, '/left_joint_trajectory_controller/follow_joint_trajectory')
        # 오른쪽 팔 궤적 제어
        self.right_arm_client = ActionClient(self, FollowJointTrajectory, '/right_joint_trajectory_controller/follow_joint_trajectory')
        # 왼쪽 그리퍼(집게) 제어
        self.left_gripper_client = ActionClient(self, GripperCommand, '/left_gripper_controller/gripper_cmd')
        # 오른쪽 그리퍼(집게) 제어
        self.right_gripper_client = ActionClient(self, GripperCommand, '/right_gripper_controller/gripper_cmd')

        # --- 구독(Subscription) 설정 ---
        # 지정된 토픽으로부터 관절 상태(JointState) 메시지가 올 때마다 joint_state_callback 함수를 실행합니다.
        self.subscription = self.create_subscription(
            JointState,
            topic,
            self.joint_state_callback,
            10  # 큐(Queue) 사이즈: 메시지가 밀릴 경우 최대 10개까지 대기
        )

        # --- 상태 추적 변수 ---
        # 로봇이 불필요하게 계속 움직이는 것을 방지하기 위해 마지막으로 보낸 위치를 기억합니다.
        self.last_arm_positions = {'left': None, 'right': None}
        self.last_gripper_positions = {'left': None, 'right': None}
        self.threshold = 0.001 # 0.001 라디안(약 0.05도) 이상의 변화가 있을 때만 명령 전송

        self.get_logger().info(f'Bimanual JointState Converter started. Subscribing to {topic}')

    def joint_state_callback(self, msg):
        """
        관절 상태 메시지가 들어오면 실행되는 함수입니다.
        데이터를 분석해서 팔과 그리퍼 명령으로 나눠 보냅니다.
        """
        # 메시지 안의 [이름 리스트]와 [위치 리스트]를 짝지어 딕셔너리로 만듭니다.
        # 예: {'joint1': 0.1, 'joint2': 0.5, ...}
        joint_data = dict(zip(msg.name, msg.position))

        # 1. 왼쪽 팔 제어 (joint1 ~ joint7)
        left_arm_joints = [f'openarm_left_joint{i}' for i in range(1, 8)]
        # 모든 관절 이름이 메시지에 포함되어 있는지 확인합니다.
        if all(j in joint_data for j in left_arm_joints):
            pos = [joint_data[j] for j in left_arm_joints]
            # 이전 위치와 비교해서 충분히 변했는지 확인합니다.
            if self.should_send_goal('left', pos):
                self.send_arm_goal(self.left_arm_client, left_arm_joints, pos)
                self.last_arm_positions['left'] = pos

        # 2. 오른쪽 팔 제어 (joint1 ~ joint7)
        right_arm_joints = [f'openarm_right_joint{i}' for i in range(1, 8)]
        if all(j in joint_data for j in right_arm_joints):
            pos = [joint_data[j] for j in right_arm_joints]
            if self.should_send_goal('right', pos):
                self.send_arm_goal(self.right_arm_client, right_arm_joints, pos)
                self.last_arm_positions['right'] = pos

        # 3. 그리퍼(집게) 제어
        for side in ['left', 'right']:
            client = self.left_gripper_client if side == 'left' else self.right_gripper_client
            # 그리퍼 관절 이름은 여러가지 가능성이 있으므로 순차적으로 찾습니다.
            gripper_joint = None
            for name in [f'openarm_{side}_finger_joint1', f'openarm_{side}_hand']:
                if name in joint_data:
                    gripper_joint = name
                    break
            
            if gripper_joint:
                pos = joint_data[gripper_joint]
                # 변화량이 기준치 이상일 때만 명령을 보냅니다.
                if self.should_send_gripper_goal(side, pos):
                    self.send_gripper_goal(client, pos)
                    self.last_gripper_positions[side] = pos

    def should_send_goal(self, side, new_pos):
        """이전 위치와 비교하여 명령을 새로 보낼지 결정합니다. (팔용)"""
        last_pos = self.last_arm_positions[side]
        if last_pos is None: return True # 처음에는 무조건 보냄
        # 하나라도 기준치(threshold) 이상 변했으면 True 반환
        return any(abs(n - l) > self.threshold for n, l in zip(new_pos, last_pos))

    def should_send_gripper_goal(self, side, new_pos):
        """이전 위치와 비교하여 명령을 새로 보낼지 결정합니다. (그리퍼용)"""
        last_pos = self.last_gripper_positions[side]
        if last_pos is None: return True
        return abs(new_pos - last_pos) > self.threshold

    def send_arm_goal(self, client, joint_names, positions):
        """팔 제어 액션 서버에 최종 목표를 전송합니다."""
        # 액션 서버가 켜져 있는지 아주 짧게(0.01초) 확인합니다.
        if not client.wait_for_server(timeout_sec=0.01):
            return

        # 액션 목표 메시지 생성
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = joint_names
        
        # 목표 지점 설정 (현재 들어온 위치와 도달 시간)
        point = JointTrajectoryPoint()
        point.positions = positions
        duration = self.get_parameter('execution_time').get_parameter_value().double_value
        point.time_from_start = rclpy.duration.Duration(seconds=duration).to_msg()
        
        goal_msg.trajectory.points = [point]
        # 비동기(Async) 방식으로 전송하여 노드가 멈추지 않게 합니다.
        client.send_goal_async(goal_msg)

    def send_gripper_goal(self, client, position):
        """그리퍼 제어 액션 서버에 최종 목표를 전송합니다."""
        if not client.wait_for_server(timeout_sec=0.01):
            return

        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = position # 벌림 정도 (단위: 미터 또는 라디안)
        goal_msg.command.max_effort = 1.0     # 쥐는 최대 힘 (1.0 = 100%)
        client.send_goal_async(goal_msg)

def main(args=None):
    """프로그램의 시작점입니다."""
    rclpy.init(args=args) # ROS2 파이썬 통신 시작
    node = BimanualJointStateConverter() # 노드 객체 생성
    try:
        rclpy.spin(node) # 프로그램이 종료될 때까지 계속 실행하며 메시지를 처리함
    except KeyboardInterrupt:
        pass # Ctrl+C를 누르면 안전하게 종료 준비
    finally:
        node.destroy_node() # 노드 파괴
        rclpy.shutdown() # ROS2 통신 종료

if __name__ == '__main__':
    main() # 스크립트 실행 시 main 함수 호출
