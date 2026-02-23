# Isaac Sim 및 관련 라이브러리 임포트
import os
import math
import numpy as np
import traceback

import omni.ext  # 엔비디아 옴니버스 확장 프로그램 라이브러리
import omni.ui as ui  # 옴니버스 사용자 인터페이스(UI) 제작 라이브러리
import omni.timeline  # 타임라인(시뮬레이션 시간) 제어
import omni.kit.app
import omni.usd  # USD(Universal Scene Description) 데이터 처리
from isaacsim.core.utils.rotations import euler_angles_to_quat, matrix_to_euler_angles
from pxr import UsdGeom, Gf, Usd  # USD 지오메트리 및 기본 클래스

from omni.isaac.dynamic_control import _dynamic_control  # 로봇 물리 제어용 인터페이스
from isaacsim.core.prims import SingleArticulation  # 관절 구조물(로봇) 핵심 모듈
from isaacsim.core.utils.types import ArticulationAction  # 로봇 제어 명령 형식

from isaacsim.robot_motion.motion_generation.lula.kinematics import LulaKinematicsSolver
from isaacsim.robot_motion.motion_generation.articulation_kinematics_solver import ArticulationKinematicsSolver


# =========================
# DOF(Degree Of Freedom) 조인트 설정
# =========================
# 화면(UI)에 표시할 관절 리스트입니다.
VISIBLE_DOFS = [
    # 왼쪽 팔
    "openarm_left_joint1",
    "openarm_left_joint2",
    "openarm_left_joint3",
    "openarm_left_joint4",
    "openarm_left_joint5",
    "openarm_left_joint6",
    "openarm_left_joint7",
    "openarm_left_finger_joint1",

    # 오른쪽 팔
    "openarm_right_joint1",
    "openarm_right_joint2",
    "openarm_right_joint3",
    "openarm_right_joint4",
    "openarm_right_joint5",
    "openarm_right_joint6",
    "openarm_right_joint7",
    "openarm_right_finger_joint1",
]

LEFT_JOINTS = [
    "openarm_left_joint1",
    "openarm_left_joint2",
    "openarm_left_joint3",
    "openarm_left_joint4",
    "openarm_left_joint5",
    "openarm_left_joint6",
    "openarm_left_joint7",
]

RIGHT_JOINTS = [
    "openarm_right_joint1",
    "openarm_right_joint2",
    "openarm_right_joint3",
    "openarm_right_joint4",
    "openarm_right_joint5",
    "openarm_right_joint6",
    "openarm_right_joint7",
]

JOINT_LIMIT_DEG = {
    "openarm_left_joint1": (-200, 80),
    "openarm_left_joint2": (-190, 10),
    "openarm_left_joint3": (-90, 90),
    "openarm_left_joint4": (0, 140),
    "openarm_left_joint5": (-90, 90),
    "openarm_left_joint6": (-45, 45),
    "openarm_left_joint7": (-90, 90),

    "openarm_right_joint1": (-80, 200),
    "openarm_right_joint2": (-10, 190),
    "openarm_right_joint3": (-90, 90),
    "openarm_right_joint4": (0, 140),
    "openarm_right_joint5": (-90, 90),
    "openarm_right_joint6": (-45, 45),
    "openarm_right_joint7": (-90, 90),
}

USE_GRIPPER_RAW = True
GRIPPER_RAW_LIMIT = {
    "openarm_left_finger_joint1": (0.0, 0.04),
    "openarm_right_finger_joint1": (0.0, 0.04),
}


# =========================
# 색상 팔레트
# =========================
CLR_BG_DARK     = 0xFF1A1A2E
CLR_BG_CARD     = 0xFF16213E
CLR_ACCENT      = 0xFF0F3460
CLR_HIGHLIGHT   = 0xFF533483
CLR_TEXT         = 0xFFE0E0E0
CLR_TEXT_DIM     = 0xFF8A8A9A
CLR_SUCCESS      = 0xFF00C853
CLR_WARNING      = 0xFFFF9100
CLR_LEFT         = 0xFF4FC3F7
CLR_RIGHT        = 0xFFFF8A65


def fmt1(x: float) -> str:
    return f"{x:.2f}"


def is_left(name: str) -> bool:
    return "_left_" in name


def is_right(name: str) -> bool:
    return "_right_" in name


def is_gripper(name: str) -> bool:
    return "finger" in name.lower()


def quat_normalize_wxyz(q):
    q = np.array(q, dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def rpy_to_quat_wxyz(roll_deg, pitch_deg, yaw_deg):
    """
    Roll(X), Pitch(Y), Yaw(Z) Euler angles (degrees) -> quaternion (w,x,y,z)
    """
    q = euler_angles_to_quat(np.array([roll_deg, pitch_deg, yaw_deg]), degrees=True)
    return q


class OpenArmAutoController(omni.ext.IExt):
    """
    OpenArm Controller 클래스:
    - JOINT 모드: 사용자가 슬라이더로 각 관절을 직접 제어합니다.
    - LULA_IK 모드: 목표 위치(큐브)를 설정하면 기구학 솔버가 관절각을 계산합니다.
    """

    def on_startup(self, ext_id):
        """
        확장 프로그램이 처음 시작(로드)될 때 호출되는 초기화 함수입니다.
        """
        print("=== Startup: OpenArm UI Control Panel ===")

        # 이전 윈도우가 열려있다면 정리합니다.
        if getattr(self, "window", None) is not None:
            try:
                self.window.destroy()
            except Exception:
                pass
            self.window = None

        self._ext_id = ext_id

        # 1. 로봇 USD 파일을 자동으로 불러옵니다.
        self._auto_open_stage(ext_id)

        # 로봇이 위치한 최상위 경로 설정
        self.robot_root_prim = "/World/openarm"

        # 2. 로봇 제어 인터페이스 초기화
        self.dc = _dynamic_control.acquire_dynamic_control_interface() # 물리 제어용
        self._bound = False # 로봇과 연결되었는지 여부
        self.art = 0  # 로봇 핸들 ID

        # IK(Inversed Kinematics) 연동 준비
        self.articulation = None  # 관절 구조체 객체
        self._articulation_ready = False

        # Lula 솔버 (ISAAC SIM의 차세대 운동학 엔진)
        self.lula_left = None
        self.lula_right = None
        self.artik_left = None
        self.artik_right = None
        self._lula_ready = False
        self._lula_failed = False

        # 끝단(EE: End-Effector)의 조인트 이름 정의
        self.ee_left = "openarm_left_ee_tcp"
        self.ee_right = "openarm_right_ee_tcp"

        # 데이터 저장을 위한 변수들
        self.targets = {} # 조인트별 목표 위치값
        self.slider_by_dof = {} # 화면의 슬라이더 객체 저장
        self.dof_name_by_handle = {} # 핸들에서 이름 찾기
        self.dof_handle_by_name = {} # 이름에서 핸들 찾기
        self.dof_index_by_name = {}  # 이름에서 인덱스(0,1,2...) 찾기 [추가]
        self.cur_tgt_labels = {} # 현재/목표 값을 표시하는 텍스트 라벨

        self._active_tab = "left" # 현재 보고 있는 탭 (왼쪽/오른쪽)

        # 제어 모드 설정: "JOINT"(직접 조작) 또는 "LULA_IK"(목표 좌표 따라가기)
        self.mode = "JOINT" 

        # IK 모드에서 사용할 목표(큐브)의 경로
        self.left_target_prim = "/World/Targets/Left"
        self.right_target_prim = "/World/Targets/Right"

        # 목표 좌표(XYZ) 초기값
        self._tgtL = {"x": 0.0, "y": 0.1535, "z": 0.0820}
        self._tgtR = {"x": 0.0, "y": -0.1535, "z": 0.0820}

        # 목표 방향(Roll, Pitch, Yaw) 초기값
        self._rpyL = {"roll": 180.0, "pitch": 0.0, "yaw": 0.0}
        self._rpyR = {"roll": 180.0, "pitch": 0.0, "yaw": 0.0}

        # 방향성 제어 활성화 여부 플래그
        self._use_oriL = False
        self._use_oriR = False

        # UI 슬라이더 참조를 위한 저장소
        self._tgt_sliders_L = {}
        self._tgt_sliders_R = {}
        self._rpy_sliders_L = {}
        self._rpy_sliders_R = {}
        self._grip_sliders_IK = {} # IK 모드용 그리퍼 슬라이더 추가 [복구]

        # 데이터 동기화 및 상태 관리 변수들
        self._is_syncing = False
        self._last_status = None
        self._printed_tb = False

        self._last_cube_poseL = None # 큐브 움직임 감지용
        self._last_cube_poseR = None

        # 3. 사용자 인터페이스(UI)를 만듭니다.
        self._build_ui()

        # 4. 시뮬레이션 매 프레임(Update)마다 실행될 함수를 등록합니다.
        app = omni.kit.app.get_app()
        self._sub = app.get_update_event_stream().create_subscription_to_pop(self._on_update)

        self._set_status("Press Play to start")

    # =========================
    # USD 파일 자동 로드
    # =========================
    def _auto_open_stage(self, ext_id: str):
        ext_mgr = omni.kit.app.get_app().get_extension_manager()
        ext_root = ext_mgr.get_extension_path(ext_id)
        usd_path = os.path.join(ext_root, "openarm_UI", "assets", "usd", "openarm_sim.usd").replace("\\", "/")

        if not os.path.exists(usd_path):
            self._set_status(f"USD not found: {usd_path}")
            return

        ctx = omni.usd.get_context()

        # 이미 해당 파일이 열려있으면 중복 로드 방지
        try:
            cur_url = ctx.get_stage_url()
            if cur_url and cur_url.endswith("openarm_sim.usd"):
                print("[OpenArmAutoController] stage already opened:", cur_url)
                return
        except Exception:
            pass

        print("[OpenArmAutoController] open_stage:", usd_path)
        ctx.open_stage(usd_path)

        self._bound = False
        self.art = 0
        self.articulation = None
        self._articulation_ready = False

    def _style(self):
        return {
            "title":      {"font_size": 22, "color": CLR_TEXT},
            "muted":      {"font_size": 14, "color": CLR_TEXT_DIM},
            "status":     {"font_size": 14, "color": CLR_WARNING},
            "section":    {"font_size": 16, "color": CLR_TEXT},
            "joint_name": {"font_size": 14, "color": CLR_TEXT},
            "range":      {"font_size": 12, "color": CLR_TEXT_DIM},
            "curtgt":     {"font_size": 13, "color": CLR_TEXT},
            "btn":        {"font_size": 14},
            "mode_joint": {"font_size": 14, "color": CLR_SUCCESS},
            "mode_ik":    {"font_size": 14, "color": CLR_HIGHLIGHT},
        }

    # =========================
    # UI 생성
    # =========================
    def _build_ui(self):
        st = self._style()

        self.window = ui.Window(
            "OpenArm Controller",
            width=540,
            height=950,
            dockPreference=ui.DockPreference.LEFT,
        )

        with self.window.frame:
            with ui.VStack(spacing=6):

                # -- 상단 헤더 영역 --
                with ui.ZStack(height=60):
                    ui.Rectangle(style={"background_color": CLR_ACCENT, "border_radius": 8})
                    with ui.VStack(spacing=2):
                        ui.Spacer(height=8)
                        with ui.HStack():
                            ui.Spacer(width=12)
                            ui.Label("OpenArm Controller", style=st["title"])
                        with ui.HStack():
                            ui.Spacer(width=12)
                            self.status = ui.Label("", style=st["status"])

                ui.Spacer(height=2)

                # -- 조작 버튼 바 --
                with ui.HStack(height=32, spacing=6):
                    ui.Button("Zero", width=70, clicked_fn=self._on_zero_position,
                               style={"font_size": 13})
                    self.mode_btn = ui.Button("Mode: JOINT", clicked_fn=self._toggle_mode,
                                              style={"font_size": 13, "color": CLR_SUCCESS})
                    ui.Spacer()
                    self.tab_left_btn = ui.Button("Left", width=65, clicked_fn=lambda: self._show_tab("left"),
                                                   style={"font_size": 13, "color": CLR_LEFT})
                    self.tab_right_btn = ui.Button("Right", width=65, clicked_fn=lambda: self._show_tab("right"),
                                                    style={"font_size": 13, "color": CLR_RIGHT})

                # -- 메인 콘텐츠 영역 --
                with ui.ZStack():
                    # LULA_IK 모드 화면
                    self.mode_ik_container = ui.VStack(visible=False, spacing=6)
                    with self.mode_ik_container:
                        with ui.ScrollingFrame(height=800):
                            with ui.VStack(spacing=8):
                                self._build_target_ui()
                    
                    # JOINT 모드 화면
                    self.mode_joint_container = ui.VStack(visible=True, spacing=6)
                    with self.mode_joint_container:
                        with ui.ScrollingFrame(height=800):
                            with ui.VStack(spacing=8):
                                self.left_container = ui.VStack(spacing=4)
                                self.right_container = ui.VStack(spacing=4)

        self._show_tab(self._active_tab)
        self._apply_mode_ui()

    def _build_target_ui(self):
        st = self._style()

        def slider_row(label, vmin, vmax, init, on_change, color=CLR_TEXT):
            with ui.HStack(height=24, spacing=6):
                ui.Label(label, width=20, style={"font_size": 14, "color": color})
                s = ui.FloatSlider(min=vmin, max=vmax, style={"font_size": 12})
                s.model.set_value(float(init))
                s.model.add_value_changed_fn(lambda m: on_change(float(m.get_value_as_float())))
            return s

        with ui.VStack(spacing=15):
            ui.Label("In LULA_IK mode, arms follow target position and orientation.", style=st["muted"], height=20)

            # --- 왼쪽 목표 지점 카드 ---
            with ui.ZStack(height=270):
                ui.Rectangle(style={"background_color": 0x40151515, "border_radius": 10})
                with ui.VStack(spacing=2):
                    ui.Spacer(height=10)
                    with ui.HStack(height=24):
                        ui.Spacer(width=15)
                        ui.Label("Left Arm Target", style={"font_size": 15, "color": CLR_LEFT})
                        ui.Spacer(width=15)
                    
                    with ui.HStack(height=24):
                        ui.Spacer(width=15)
                        ui.Label("Position Targets", style=st["muted"], width=120)
                        ui.Spacer()
                        ui.Label("Orient", style=st["muted"], width=50)
                        self._cb_oriL = ui.CheckBox(width=20, style={"color": 0xFF000000})
                        self._cb_oriL.model.set_value(self._use_oriL)
                        self._cb_oriL.model.add_value_changed_fn(lambda m: self._on_toggle_ori("L", m.get_value_as_bool()))
                        ui.Spacer(width=15)

                    with ui.HStack(height=72): # 3 * 24
                        ui.Spacer(width=15)
                        with ui.VStack(spacing=0):
                            self._tgt_sliders_L["x"] = slider_row("X", -1.0, 1.0, self._tgtL["x"], lambda v: self._on_target_slider("L", "x", v), CLR_LEFT)
                            self._tgt_sliders_L["y"] = slider_row("Y", -1.0, 1.0, self._tgtL["y"], lambda v: self._on_target_slider("L", "y", v), CLR_LEFT)
                            self._tgt_sliders_L["z"] = slider_row("Z", 0.0, 1.5, self._tgtL["z"], lambda v: self._on_target_slider("L", "z", v), CLR_LEFT)
                        ui.Spacer(width=15)

                    ui.Spacer(height=6)

                    with ui.HStack(height=20):
                        ui.Spacer(width=15)
                        ui.Label("Orientation (RPY Degrees)", style=st["muted"])
                        ui.Spacer(width=15)

                    with ui.HStack(height=72):
                        ui.Spacer(width=15)
                        with ui.VStack(spacing=0):
                            self._rpy_sliders_L["roll"] = slider_row("R", -180.0, 180.0, self._rpyL["roll"], lambda v: self._on_target_rpy_slider("L", "roll", v), CLR_LEFT)
                            self._rpy_sliders_L["pitch"] = slider_row("P", -90.0, 90.0, self._rpyL["pitch"], lambda v: self._on_target_rpy_slider("L", "pitch", v), CLR_LEFT)
                            self._rpy_sliders_L["yaw"] = slider_row("Y", -180.0, 180.0, self._rpyL["yaw"], lambda v: self._on_target_rpy_slider("L", "yaw", v), CLR_LEFT)
                        ui.Spacer(width=15)

                    # [추가] IK 모드에서도 그리퍼 슬라이더를 표시합니다. (이름 기반으로 상시 생성)
                    with ui.HStack(height=24):
                        ui.Spacer(width=15)
                        left_grip_name = "openarm_left_finger_joint1"
                        self._grip_sliders_IK["L"] = slider_row("G", 0.0, 0.04, 0.0, 
                            lambda v: self._on_joint_slider(left_grip_name, v), CLR_SUCCESS)
                        ui.Spacer(width=15)
                    ui.Spacer()

            # --- 오른쪽 목표 지점 카드 ---
            with ui.ZStack(height=270):
                ui.Rectangle(style={"background_color": 0x40151515, "border_radius": 10})
                with ui.VStack(spacing=2):
                    ui.Spacer(height=10)
                    with ui.HStack(height=24):
                        ui.Spacer(width=15)
                        ui.Label("Right Arm Target", style={"font_size": 15, "color": CLR_RIGHT})
                        ui.Spacer(width=15)
                    
                    with ui.HStack(height=24):
                        ui.Spacer(width=15)
                        ui.Label("Position Targets", style=st["muted"], width=120)
                        ui.Spacer()
                        ui.Label("Orient", style=st["muted"], width=50)
                        self._cb_oriR = ui.CheckBox(width=20, style={"color": 0xFF000000})
                        self._cb_oriR.model.set_value(self._use_oriR)
                        self._cb_oriR.model.add_value_changed_fn(lambda m: self._on_toggle_ori("R", m.get_value_as_bool()))
                        ui.Spacer(width=15)

                    with ui.HStack(height=72):
                        ui.Spacer(width=15)
                        with ui.VStack(spacing=0):
                            self._tgt_sliders_R["x"] = slider_row("X", -1.0, 1.0, self._tgtR["x"], lambda v: self._on_target_slider("R", "x", v), CLR_RIGHT)
                            self._tgt_sliders_R["y"] = slider_row("Y", -1.0, 1.0, self._tgtR["y"], lambda v: self._on_target_slider("R", "y", v), CLR_RIGHT)
                            self._tgt_sliders_R["z"] = slider_row("Z", 0.0, 1.5, self._tgtR["z"], lambda v: self._on_target_slider("R", "z", v), CLR_RIGHT)
                        ui.Spacer(width=15)

                    ui.Spacer(height=6)

                    with ui.HStack(height=20):
                        ui.Spacer(width=15)
                        ui.Label("Orientation (RPY Degrees)", style=st["muted"])
                        ui.Spacer(width=15)

                    with ui.HStack(height=72):
                        ui.Spacer(width=15)
                        with ui.VStack(spacing=0):
                            self._rpy_sliders_R["roll"] = slider_row("R", -180.0, 180.0, self._rpyR["roll"], lambda v: self._on_target_rpy_slider("R", "roll", v), CLR_RIGHT)
                            self._rpy_sliders_R["pitch"] = slider_row("P", -90.0, 90.0, self._rpyR["pitch"], lambda v: self._on_target_rpy_slider("R", "pitch", v), CLR_RIGHT)
                            self._rpy_sliders_R["yaw"] = slider_row("Y", -180.0, 180.0, self._rpyR["yaw"], lambda v: self._on_target_rpy_slider("R", "yaw", v), CLR_RIGHT)
                        ui.Spacer(width=15)
                    
                    # [추가] IK 모드에서도 그리퍼 슬라이더를 표시합니다. (이름 기반으로 상시 생성)
                    with ui.HStack(height=24):
                        ui.Spacer(width=15)
                        right_grip_name = "openarm_right_finger_joint1"
                        self._grip_sliders_IK["R"] = slider_row("G", 0.0, 0.04, 0.0, 
                            lambda v: self._on_joint_slider(right_grip_name, v), CLR_SUCCESS)
                        ui.Spacer(width=15)
                    ui.Spacer()

            ui.Spacer(height=5)

            with ui.HStack(height=28, spacing=6):
                ui.Button("Reset Targets", clicked_fn=self._reset_targets,
                           style={"font_size": 13})
                ui.Button("Set both to current", clicked_fn=self._reset_to_current_all,
                           style={"font_size": 13})
            
            with ui.HStack(height=28, spacing=6):
                ui.Button("Set left to current", clicked_fn=lambda: self._set_target_to_current("L"),
                           style={"font_size": 12})
                ui.Button("Set right to current", clicked_fn=lambda: self._set_target_to_current("R"),
                           style={"font_size": 12})
            
            with ui.HStack(height=28, spacing=6):
                ui.Button("Reset Orientation", clicked_fn=self._reset_all_rpy,
                           style={"font_size": 12})
                ui.Button("Sync UI -> Prim", clicked_fn=self._sync_targets_to_stage,
                           style={"font_size": 12})
                ui.Spacer()

    def _section_header(self, title: str, color=CLR_TEXT):
        with ui.ZStack(height=28):
            ui.Rectangle(style={"background_color": CLR_ACCENT, "border_radius": 4})
            with ui.HStack():
                ui.Spacer(width=8)
                ui.Label(title, style={"font_size": 15, "color": color})

    def _show_tab(self, name: str):
        self._active_tab = name
        if getattr(self, "left_container", None) is None or getattr(self, "right_container", None) is None:
            return
        try:
            self.left_container.visible = (name == "left")
            self.right_container.visible = (name == "right")
        except Exception:
            pass

        try:
            if name == "left":
                self.tab_left_btn.style = {"font_size": 13, "color": CLR_LEFT}
                self.tab_right_btn.style = {"font_size": 13, "color": CLR_TEXT_DIM}
            else:
                self.tab_left_btn.style = {"font_size": 13, "color": CLR_TEXT_DIM}
                self.tab_right_btn.style = {"font_size": 13, "color": CLR_RIGHT}
        except Exception:
            pass

    def _set_status(self, s: str):
        if self._last_status == s:
            return
        self._last_status = s
        if getattr(self, "status", None) is not None:
            self.status.text = s
        print("[OpenArmAutoController]", s)

    def _apply_mode_ui(self):
        is_joint = (self.mode == "JOINT")
        try:
            self.mode_joint_container.visible = is_joint
            self.mode_ik_container.visible = not is_joint
        except Exception:
            pass

        try:
            if is_joint:
                self.mode_btn.text = "Mode: JOINT"
                self.mode_btn.style = {"font_size": 13, "color": CLR_SUCCESS}
            else:
                self.mode_btn.text = "Mode: LULA_IK"
                self.mode_btn.style = {"font_size": 13, "color": CLR_HIGHLIGHT}
        except Exception:
            pass

    def _toggle_mode(self):
        # 모드 전환
        self.mode = "LULA_IK" if self.mode == "JOINT" else "JOINT"
        self._apply_mode_ui()

        stage = omni.usd.get_context().get_stage()
        if stage is not None:
            # 1. 시뮬레이션 물리 모델과 동기화하기 위한 핸들 등이 필요할 수 있으므로 강제 확인
            self._ensure_target_prims(stage)

            # 2. 전환되는 모드에 따라 상태를 동기화합니다.
            if self.mode == "JOINT":
                # LULA_IK -> JOINT: 현재 로봇의 실제 관절각을 읽어와서 슬라이더와 목표값에 적용합니다.
                if self._bound:
                    self._is_syncing = True # 슬라이더 변경 시 발생할 수 있는 콜백 재입력 방지
                    try:
                        for dof, slider in self.slider_by_dof.items():
                            name = self.dof_name_by_handle.get(dof, "")
                            grip = (USE_GRIPPER_RAW and is_gripper(name))
                            cur_pos = float(self.dc.get_dof_position(dof))
                            
                            # UI 값으로 변환 (deg/rad/raw)
                            val_ui = cur_pos if grip else math.degrees(cur_pos)
                            
                            # 슬라이더 모델 업데이트
                            slider.model.set_value(val_ui)
                            # 내부 타겟 변수 업데이트
                            self.targets[dof] = cur_pos
                    finally:
                        self._is_syncing = False
                self._set_status("Switched to JOINT: Synced sliders to current robot state.")

            elif self.mode == "LULA_IK":
                # JOINT -> LULA_IK: 현재 로봇의 끝단(EE) 위치로 목표 큐브들을 이동시키고, 그리퍼 슬라이더를 동기화합니다.
                if self._lula_ready:
                    self._set_target_to_current("L")
                    self._set_target_to_current("R")
                
                # IK 모드의 전용 그리퍼 슬라이더(G) 동기화
                self._is_syncing = True
                try:
                    for side, slider in self._grip_sliders_IK.items():
                        name = "openarm_left_finger_joint1" if side == "L" else "openarm_right_finger_joint1"
                        handle = self.dof_handle_by_name.get(name)
                        if handle is not None:
                            cur_pos = self.targets.get(handle, 0.0)
                            slider.model.set_value(float(cur_pos))
                finally:
                    self._is_syncing = False

                self._set_status("Switched to LULA_IK: Synced targets and grippers.")

            # 3. 변경 사항을 3D 화면에 즉시 반영
            self._sync_targets_to_stage()

        print(f"[OpenArmAutoController] Mode toggled to: {self.mode}")

    # ---------------- Update 루프 ----------------
    def _on_update(self, e):
        """
        시뮬레이션이 돌아가는 동안 매 순간(프레임) 실행되는 메인 루프입니다.
        """
        timeline = omni.timeline.get_timeline_interface()
        is_playing = timeline.is_playing()

        # 시뮬레이션이 중지되었다면 하드웨어 연결 정보를 초기화합니다.
        if not is_playing and self._bound:
            self._bound = False
            self._articulation_ready = False
            self._lula_ready = False
            self.articulation = None
            self.artik_left = None
            self.artik_right = None
            self._set_status("Simulation stopped. Ready to re-bind.")

        # 로봇과 성공적으로 연결된 상태라면 (시뮬레이션 실행 중)
        if self._bound:
            stage = omni.usd.get_context().get_stage()
            if stage is not None:
                # 3D 화면 요소(큐브 등)가 없으면 생성합니다.
                self._ensure_target_prims(stage)
                
                # [중요] 3D 화면에서 직접 큐브를 움직였는지 확인하고 UI를 갱신합니다.
                self._sync_ui_from_viewport(stage)

            # IK(기구학) 모드인 경우: 목표 큐브를 향해 로봇이 움직이도록 계산하여 명령을 내립니다.
            if self.mode == "LULA_IK" and self._lula_ready and self._articulation_ready:
                try:
                    self._step_lula_ik(stage)
                except Exception:
                    if not self._printed_tb:
                        self._printed_tb = True
                        print("=== IK Status Error ===")
                        print(traceback.format_exc())

            # 직접 운전(JOINT) 모드인 경우: 슬라이더 값을 물리 엔진에 직접 전달합니다.
            if self.mode == "JOINT":
                try:
                    self.dc.wake_up_articulation(self.art) # 물리 엔진 활성화
                except Exception:
                    pass

                for dof, tgt in self.targets.items():
                    try:
                        # 물리 엔진의 각 조인트에 목표 위치값을 입력합니다.
                        self.dc.set_dof_position_target(dof, float(tgt))
                    except Exception:
                        pass

            # 화면에 현재 로봇의 실제 위치값을 실시간으로 갱신합니다.
            self._refresh_current_labels()
            return

        # 시뮬레이션이 아직 시작되지 않았다면 대기 메시지를 표시합니다.
        if not is_playing:
            self._set_status("Press Play to activate hardware.")
            return

        # 재생 버튼을 눌렀다면, 실제 로봇(Articulation)을 찾아 연결을 시도합니다.
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            self._set_status("Stage not ready...")
            return

        # 1) 로봇 물체를 물리 엔진 핸들로 가져옵니다.
        art = self.dc.get_articulation(self.robot_root_prim)
        if art == 0:
            # 설정한 경로에 없다면 전체 씬에서 찾아봅니다.
            found_path, found_art = self._probe_articulation_under(stage, "/World")
            if found_art != 0:
                art = found_art

        if art == 0:
            self._set_status("Searching articulation...")
            return

        # 2) 찾은 로봇을 조작 준비(Bind) 상태로 만듭니다.
        self._bind(art, self.robot_root_prim)

        # 3) IK 로직과 3D 화면 요소들을 준비합니다.
        self._init_single_articulation(stage)
        self._init_lula_and_artik()
        self._ensure_target_prims(stage)
        # 처음 한 번만 UI 값을 화면에 반영합니다.
        self._sync_targets_to_stage()

    def _probe_articulation_under(self, stage, base_path: str):
        base = stage.GetPrimAtPath(base_path)
        if not base or not base.IsValid():
            return None, 0
        stack = [base]
        while stack:
            p = stack.pop()
            p_str = str(p.GetPath())
            if p_str.startswith("/World/Targets"): continue
            if "constraint" in p_str.lower() or "actiongraph" in p_str.lower(): continue
            try:
                art = self.dc.get_articulation(p_str)
            except Exception:
                art = 0
            if art != 0: return p_str, art
            for c in p.GetChildren(): stack.append(c)
        return None, 0

    def _bind(self, art_handle: int, root_path: str):
        self.art = art_handle
        self._bound = True
        self.targets.clear()
        self.slider_by_dof.clear()
        self.dof_name_by_handle.clear()
        self.cur_tgt_labels.clear()
        self._set_status(f"Bound: {root_path}")
        self._build_sliders()
        self._show_tab(self._active_tab)
        self.dof_handle_by_name = {name: dof for dof, name in self.dof_name_by_handle.items()}

    def _init_single_articulation(self, stage):
        if self._articulation_ready and self.articulation is not None: return
        candidates = [
            self.robot_root_prim, 
            f"{self.robot_root_prim}/root_joint",
            f"{self.robot_root_prim}/openarm_left_ee_tcp",
            f"{self.robot_root_prim}/openarm_right_ee_tcp",
        ]
        for prim_path in candidates:
            prim = stage.GetPrimAtPath(prim_path)
            if not prim or not prim.IsValid(): continue
            try:
                art = SingleArticulation(prim_path)
                if not art.handles_initialized: art.initialize()
                self.articulation = art
                self._articulation_ready = True
                
                # DOF 이름과 인덱스 매핑 저장
                self.dof_index_by_name = {name: i for i, name in enumerate(art.dof_names)}
                return
            except Exception: continue

    def _init_lula_and_artik(self):
        if self._lula_ready or self._lula_failed: return
        if not self._articulation_ready or self.articulation is None: return
        ext_mgr = omni.kit.app.get_app().get_extension_manager()
        ext_root = ext_mgr.get_extension_path(self._ext_id)
        assets = os.path.join(ext_root, "openarm_UI", "assets")
        urdf_path = os.path.join(assets, "openarm_bimanual.urdf")
        left_yaml = os.path.join(assets, "openarm_left.yaml")
        right_yaml = os.path.join(assets, "openarm_right.yaml")
        try:
            self.lula_left = LulaKinematicsSolver(left_yaml, urdf_path)
            self.lula_right = LulaKinematicsSolver(right_yaml, urdf_path)
            self.artik_left = ArticulationKinematicsSolver(self.articulation, self.lula_left, self.ee_left)
            self.artik_right = ArticulationKinematicsSolver(self.articulation, self.lula_right, self.ee_right)
            self._lula_ready = True
            self._set_target_to_current("L"); self._set_target_to_current("R")
        except Exception as e:
            self._lula_failed = True
            self._set_status(f"IK Init Failed: {e}")

    # ---------------- Target Prim (3D Cube) ----------------
    def _ensure_target_prims(self, stage):
        targets_root = "/World/Targets"
        if not stage.GetPrimAtPath(targets_root).IsValid():
            stage.DefinePrim(targets_root, "Xform")

        def ensure_target(path_xform: str, default_translate: Gf.Vec3d, default_quat_wxyz):
            prim = stage.GetPrimAtPath(path_xform)
            if not prim or not prim.IsValid():
                prim = stage.DefinePrim(path_xform, "Xform")
            xf = UsdGeom.Xformable(prim)
            t_op = None
            for op in xf.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate: t_op = op; break
            if t_op is None: t_op = xf.AddTranslateOp()
            if not t_op.GetAttr().HasAuthoredValueOpinion(): t_op.Set(default_translate)
            o_op = None
            for op in xf.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeOrient: o_op = op; break
            if o_op is None: o_op = xf.AddOrientOp()
            w, x, y, z = default_quat_wxyz
            q = Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z)))
            if not o_op.GetAttr().HasAuthoredValueOpinion(): o_op.Set(q)

        q_def = rpy_to_quat_wxyz(180, 0, 0)
        ensure_target(self.left_target_prim, Gf.Vec3d(self._tgtL["x"], self._tgtL["y"], self._tgtL["z"]), q_def)
        ensure_target(self.right_target_prim, Gf.Vec3d(self._tgtR["x"], self._tgtR["y"], self._tgtR["z"]), q_def)

    def _set_target_translation(self, stage, prim_path: str, xyz):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid(): return
        xf = UsdGeom.Xformable(prim)
        op = None
        for xop in xf.GetOrderedXformOps():
            if xop.GetOpType() == UsdGeom.XformOp.TypeTranslate: op = xop; break
        if op is None: op = xf.AddTranslateOp()
        op.Set(Gf.Vec3d(float(xyz[0]), float(xyz[1]), float(xyz[2])))

    def _set_target_orientation_wxyz(self, stage, prim_path: str, quat_wxyz):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid(): return
        xf = UsdGeom.Xformable(prim)
        op = None
        for xop in xf.GetOrderedXformOps():
            if xop.GetOpType() == UsdGeom.XformOp.TypeOrient: op = xop; break
        if op is None: op = xf.AddOrientOp()
        w, x, y, z = quat_wxyz
        op.Set(Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z))))

    def _get_world_translation(self, stage, prim_path: str):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid(): return None
        xf = UsdGeom.Xformable(prim)
        M = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        t = M.ExtractTranslation()
        return (t[0], t[1], t[2])

    def _get_world_quat_wxyz(self, stage, prim_path: str):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid(): return None
        xf = UsdGeom.Xformable(prim)
        M = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        r = M.ExtractRotationQuat()
        return (r.GetReal(), r.GetImaginary()[0], r.GetImaginary()[1], r.GetImaginary()[2])

    # ---------------- UI Events ----------------
    def _on_target_slider(self, side: str, axis: str, value: float):
        if self._is_syncing: return
        if side == "L": self._tgtL[axis] = float(value)
        else: self._tgtR[axis] = float(value)
        self._sync_targets_to_stage()

    def _on_target_rpy_slider(self, side: str, axis: str, value: float):
        if self._is_syncing: return
        if side == "L": self._rpyL[axis] = float(value)
        else: self._rpyR[axis] = float(value)
        self._sync_targets_to_stage()

    def _on_toggle_ori(self, side: str, value: bool):
        if side == "L": self._use_oriL = value
        else: self._use_oriR = value

    def _sync_ui_from_viewport(self, stage):
        """3D 화면에서 큐브를 직접 움직였을 경우 UI 값을 그에 맞게 업데이트합니다."""
        if self._is_syncing: return

        def check_and_sync(side: str, prim_path: str):
            pos = self._get_world_translation(stage, prim_path)
            quat = self._get_world_quat_wxyz(stage, prim_path)
            if pos is None or quat is None: return

            # 마지막 저장된 위치와 비교하여 변화가 있는지 확인
            last = self._last_cube_poseL if side == "L" else self._last_cube_poseR
            curr = (pos, quat)
            
            if last is not None:
                # 위치나 회전이 미세하게라도 변했는지 확인
                if np.allclose(last[0], curr[0], atol=1e-4) and np.allclose(last[1], curr[1], atol=1e-4):
                    return

            # 변화가 감지됨 -> UI 내부 값 업데이트
            if side == "L":
                self._tgtL = {"x": pos[0], "y": pos[1], "z": pos[2]}
                r, p, y = matrix_to_euler_angles(Gf.Matrix3d(Gf.Quatd(*quat)), degrees=True)
                self._rpyL = {"roll": r, "pitch": p, "yaw": y}
                self._last_cube_poseL = curr
            else:
                self._tgtR = {"x": pos[0], "y": pos[1], "z": pos[2]}
                r, p, y = matrix_to_euler_angles(Gf.Matrix3d(Gf.Quatd(*quat)), degrees=True)
                self._rpyR = {"roll": r, "pitch": p, "yaw": y}
                self._last_cube_poseR = curr

            # 실제 슬라이더 UI 갱신 (콜백 재호출 방지를 위해 루프 밖에서 처리)
            self._sync_ui_from_state()

        check_and_sync("L", self.left_target_prim)
        check_and_sync("R", self.right_target_prim)

    def _sync_targets_to_stage(self):
        stage = omni.usd.get_context().get_stage()
        if stage is None: return
        self._ensure_target_prims(stage)
        self._set_target_translation(stage, self.left_target_prim, (self._tgtL["x"], self._tgtL["y"], self._tgtL["z"]))
        self._set_target_translation(stage, self.right_target_prim, (self._tgtR["x"], self._tgtR["y"], self._tgtR["z"]))
        qL = rpy_to_quat_wxyz(self._rpyL["roll"], self._rpyL["pitch"], self._rpyL["yaw"])
        qR = rpy_to_quat_wxyz(self._rpyR["roll"], self._rpyR["pitch"], self._rpyR["yaw"])
        self._set_target_orientation_wxyz(stage, self.left_target_prim, qL)
        self._set_target_orientation_wxyz(stage, self.right_target_prim, qR)
        
        # 마지막 위치 저장 (강제 동기화 후)
        self._last_cube_poseL = (self._get_world_translation(stage, self.left_target_prim), self._get_world_quat_wxyz(stage, self.left_target_prim))
        self._last_cube_poseR = (self._get_world_translation(stage, self.right_target_prim), self._get_world_quat_wxyz(stage, self.right_target_prim))
        
        self._sync_ui_from_state()

    def _sync_ui_from_state(self):
        if self._is_syncing: return
        self._is_syncing = True
        try:
            for k in ["x", "y", "z"]:
                if k in self._tgt_sliders_L: self._tgt_sliders_L[k].model.set_value(self._tgtL[k])
                if k in self._tgt_sliders_R: self._tgt_sliders_R[k].model.set_value(self._tgtR[k])
            for k in ["roll", "pitch", "yaw"]:
                if k in self._rpy_sliders_L: self._rpy_sliders_L[k].model.set_value(self._rpyL[k])
                if k in self._rpy_sliders_R: self._rpy_sliders_R[k].model.set_value(self._rpyR[k])
        finally: self._is_syncing = False

    def _reset_targets(self):
        self._tgtL = {"x": 0.0, "y": 0.1535, "z": 0.0820}
        self._tgtR = {"x": 0.0, "y": -0.1535, "z": 0.0820}
        self._rpyL = {"roll": 180.0, "pitch": 0.0, "yaw": 0.0}
        self._rpyR = {"roll": 180.0, "pitch": 0.0, "yaw": 0.0}
        self._sync_targets_to_stage()

    def _reset_to_current_all(self):
        self._set_target_to_current("L"); self._set_target_to_current("R")

    def _set_target_to_current(self, side: str):
        if self.articulation is None or self.artik_left is None: return
        ik_solver = self.artik_left if side == "L" else self.artik_right
        try:
            pos, rot_mat = ik_solver.compute_end_effector_pose()
            euler = matrix_to_euler_angles(rot_mat, degrees=True)
            if side == "L":
                self._tgtL = {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])}
                self._rpyL = {"roll": float(euler[0]), "pitch": float(euler[1]), "yaw": float(euler[2])}
            else:
                self._tgtR = {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])}
                self._rpyR = {"roll": float(euler[0]), "pitch": float(euler[1]), "yaw": float(euler[2])}
            self._sync_targets_to_stage()
        except: pass

    def _reset_all_rpy(self):
        self._rpyL = {"roll": 180.0, "pitch": 0.0, "yaw": 0.0}
        self._rpyR = {"roll": 180.0, "pitch": 0.0, "yaw": 0.0}
        self._sync_targets_to_stage()

    # ---------------- IK 연산 및 실행 ----------------
    def _step_lula_ik(self, stage):
        if stage is None or self.articulation is None: return
        pos_root, quat_root = self.articulation.get_world_pose()
        self.lula_left.set_robot_base_pose(pos_root, quat_root)
        self.lula_right.set_robot_base_pose(pos_root, quat_root)

        tgtL = self._get_world_translation(stage, self.left_target_prim)
        tgtR = self._get_world_translation(stage, self.right_target_prim)
        qL_wxyz = self._get_world_quat_wxyz(stage, self.left_target_prim)
        qR_wxyz = self._get_world_quat_wxyz(stage, self.right_target_prim)

        if tgtL is None or tgtR is None: return
        posL = np.array(tgtL, dtype=np.float32)
        posR = np.array(tgtR, dtype=np.float32)
        oriL = quat_normalize_wxyz(qL_wxyz).astype(np.float32) if self._use_oriL else None
        oriR = quat_normalize_wxyz(qR_wxyz).astype(np.float32) if self._use_oriR else None

        actionL, succL = self.artik_left.compute_inverse_kinematics(posL, oriL)
        actionR, succR = self.artik_right.compute_inverse_kinematics(posR, oriR)

        combined_positions = []
        combined_indices = []
        
        # 1) 팔 조인트 각도 병합
        if succL:
            combined_positions.extend(actionL.joint_positions)
            combined_indices.extend(actionL.joint_indices)
        if succR:
            combined_positions.extend(actionR.joint_positions)
            combined_indices.extend(actionR.joint_indices)

        # 2) 그리퍼 조인트 각도 병합 (인덱스 매핑 사용)
        for name in ["openarm_left_finger_joint1", "openarm_right_finger_joint1"]:
            handle = self.dof_handle_by_name.get(name)
            idx = self.dof_index_by_name.get(name)
            if handle is not None and idx is not None:
                combined_positions.append(self.targets.get(handle, 0.0))
                combined_indices.append(idx)

        if combined_indices:
            merged_action = ArticulationAction(joint_positions=np.array(combined_positions), joint_indices=np.array(combined_indices))
            self.articulation.get_articulation_controller().apply_action(merged_action)

        self._set_status(f"IK Status -> Left:{'OK' if succL else 'FAIL'} Right:{'OK' if succR else 'FAIL'}")

    # ---------------- 슬라이더 생성 ----------------
    def _build_sliders(self):
        dof_count = self.dc.get_articulation_dof_count(self.art)
        name_map = {}
        for i in range(dof_count):
            dof = self.dc.get_articulation_dof(self.art, i)
            name = self.dc.get_dof_name(dof)
            cur = float(self.dc.get_dof_position(dof))
            name_map[name] = (dof, cur)
        st = self._style()
        with self.left_container:
            self._section_header("Left Arm", CLR_LEFT)
            for name in [n for n in VISIBLE_DOFS if is_left(n)]: self._add_joint_row(name, name_map, st)
        with self.right_container:
            self._section_header("Right Arm", CLR_RIGHT)
            for name in [n for n in VISIBLE_DOFS if is_right(n)]: self._add_joint_row(name, name_map, st)

    def _add_joint_row(self, name: str, name_map: dict, st: dict):
        if name not in name_map: return
        dof, cur_rad = name_map[name]
        grip = (USE_GRIPPER_RAW and is_gripper(name))
        if grip: lo, hi = GRIPPER_RAW_LIMIT.get(name, (0.0, 0.04)); cur_ui = cur_rad; unit = ""
        else: lo, hi = JOINT_LIMIT_DEG.get(name, (-180, 180)); cur_ui = math.degrees(cur_rad); unit = "deg"
        self.targets[dof] = float(cur_rad)
        short = name.split("_")[-1].upper() if not grip else "GRIP"
        with ui.VStack(spacing=2):
            with ui.HStack(height=22):
                ui.Label(short, style={"font_size": 14, "color": CLR_TEXT}, width=50)
                lbl = ui.Label(f"cur:{fmt1(cur_ui)}{unit}  tgt:{fmt1(cur_ui)}{unit}", style=st["curtgt"])
                self.cur_tgt_labels[dof] = lbl
            s = ui.FloatSlider(min=float(lo), max=float(hi), style={"font_size": 12})
            s.model.set_value(float(cur_ui))
            self.slider_by_dof[dof] = s
            self.dof_name_by_handle[dof] = name
            s.model.add_value_changed_fn(lambda m, n=name, h=dof: self._on_joint_slider(n, m, h))

    def _refresh_current_labels(self):
        for dof, slider in self.slider_by_dof.items():
            name = self.dof_name_by_handle.get(dof, "")
            grip = (USE_GRIPPER_RAW and is_gripper(name))
            cur = float(self.dc.get_dof_position(dof))
            tgt_ui = slider.model.get_value_as_float()
            cur_ui = cur if grip else math.degrees(cur)
            unit = "" if grip else "deg"
            if dof in self.cur_tgt_labels: self.cur_tgt_labels[dof].text = f"cur:{fmt1(cur_ui)}{unit}  tgt:{fmt1(tgt_ui)}{unit}"

    def _on_joint_slider(self, dof_name: str, value_ui, dof_handle: int = None):
        if not self._bound: return
        
        if dof_handle is None:
            dof_handle = self.dof_handle_by_name.get(dof_name)
        if dof_handle is None: return

        # value_ui가 모델/슬라이더/수치값인지 확인하여 값을 추출합니다.
        try:
            if hasattr(value_ui, "get_value_as_float"):
                val = float(value_ui.get_value_as_float())
            elif hasattr(value_ui, "model"):
                val = float(value_ui.model.get_value_as_float())
            else:
                val = float(value_ui)
        except Exception:
            return

        # IK 모드에서는 일반 조인트 슬라이더 무시 (그리퍼는 허용)
        if self.mode == "LULA_IK" and not is_gripper(dof_name): return

        self.targets[dof_handle] = val if is_gripper(dof_name) else math.radians(val)

    def _on_zero_position(self):
        if not self._bound: return
        for dof, slider in self.slider_by_dof.items(): slider.model.set_value(0.0)
        for dof in self.targets: self.targets[dof] = 0.0
        self._set_status("Zero Position applied")

    def on_shutdown(self):
        self._sub = None
        if self.window: self.window.destroy(); self.window = None

if __name__ == '__main__':
    pass