import os
import math
import numpy as np
import traceback

import omni.ext
import omni.ui as ui
import omni.timeline
import omni.kit.app
import omni.usd
from isaacsim.core.utils.rotations import euler_angles_to_quat, matrix_to_euler_angles
from pxr import UsdGeom, Gf, Usd

from omni.isaac.dynamic_control import _dynamic_control
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction

from isaacsim.robot_motion.motion_generation.lula.kinematics import LulaKinematicsSolver
from isaacsim.robot_motion.motion_generation.articulation_kinematics_solver import ArticulationKinematicsSolver


# =========================
# DOF Configuration
# =========================
VISIBLE_DOFS = [
    # Left arm
    "openarm_left_joint1",
    "openarm_left_joint2",
    "openarm_left_joint3",
    "openarm_left_joint4",
    "openarm_left_joint5",
    "openarm_left_joint6",
    "openarm_left_joint7",
    "openarm_left_finger_joint1",

    # Right arm
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
# Color Palette
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
    Rotation order: XYZ extrinsic = ZYX intrinsic
    """
    r = np.radians(float(roll_deg))
    p = np.radians(float(pitch_deg))
    y = np.radians(float(yaw_deg))

    cr, sr = np.cos(r / 2), np.sin(r / 2)
    cp, sp = np.cos(p / 2), np.sin(p / 2)
    cy, sy = np.cos(y / 2), np.sin(y / 2)

    # Rotation order: XYZ extrinsic = ZYX intrinsic
    # Use Isaac Sim core utility instead of custom implementation
    q = euler_angles_to_quat(np.array([roll_deg, pitch_deg, yaw_deg]), degrees=True)
    return q


class OpenArmAutoController(omni.ext.IExt):
    """
    - JOINT mode: slider -> dynamic_control dof position target
    - LULA_IK mode: target cube position -> ArticulationKinematicsSolver IK -> apply_action
    """

    def on_startup(self, ext_id):
        print("=== LOADED: openarm_UI/extension.py LULA-ARTIK-UI-2026-02-23 ===")

        # Prevent duplicate window
        if getattr(self, "window", None) is not None:
            try:
                self.window.destroy()
            except Exception:
                pass
            self.window = None

        self._ext_id = ext_id

        # Auto-load USD
        self._auto_open_stage(ext_id)

        # Robot root prim (default)
        self.robot_root_prim = "/World/openarm"

        # Dynamic Control
        self.dc = _dynamic_control.acquire_dynamic_control_interface()
        self._bound = False
        self.art = 0  # dc articulation handle

        # SingleArticulation (for IK)
        self.articulation = None  # SingleArticulation
        self._articulation_ready = False

        # Lula + ArticulationKinematicsSolver
        self.lula_left = None
        self.lula_right = None
        self.artik_left = None
        self.artik_right = None
        self._lula_ready = False
        self._lula_failed = False

        # EE frame names
        self.ee_left = "openarm_left_ee_tcp"
        self.ee_right = "openarm_right_ee_tcp"

        # dof_handle -> target rad/raw
        self.targets = {}

        # dof_handle -> ui.FloatSlider
        self.slider_by_dof = {}
        # dof_handle -> dof_name
        self.dof_name_by_handle = {}
        # dof_name -> dof_handle
        self.dof_handle_by_name = {}
        # dof_handle -> label
        self.cur_tgt_labels = {}

        self._active_tab = "left"

        # =========================
        # Mode / Targets
        # =========================
        self.mode = "JOINT"  # "JOINT" or "LULA_IK"

        self.left_target_prim = "/World/Targets/Left"
        self.right_target_prim = "/World/Targets/Right"

        # UI target value store
        self._tgtL = {"x": 0.0, "y": 0.1535, "z": 0.0689}
        self._tgtR = {"x": 0.0, "y": -0.1535, "z": 0.0689}

        # Orientation: RPY (Roll, Pitch, Yaw) in degrees
        self._rpyL = {"roll": 180.0, "pitch": 0.0, "yaw": 0.0}
        self._rpyR = {"roll": 180.0, "pitch": 0.0, "yaw": 0.0}

        # Orientation Toggle Flags
        self._use_oriL = False
        self._use_oriR = False

        # Target UI slider references
        self._tgt_sliders_L = {}
        self._tgt_sliders_R = {}
        self._rpy_sliders_L = {}
        self._rpy_sliders_R = {}

        self._is_syncing = False
        self._last_status = None
        self._printed_tb = False

        self._build_ui()

        app = omni.kit.app.get_app()
        self._sub = app.get_update_event_stream().create_subscription_to_pop(self._on_update)

        self._set_status("Press Play to start")

    # =========================
    # USD Load
    # =========================
    def _auto_open_stage(self, ext_id: str):
        ext_mgr = omni.kit.app.get_app().get_extension_manager()
        ext_root = ext_mgr.get_extension_path(ext_id)
        usd_path = os.path.join(ext_root, "openarm_UI", "assets", "usd", "openarm_sim.usd").replace("\\", "/")

        if not os.path.exists(usd_path):
            self._set_status(f"USD not found: {usd_path}")
            return

        ctx = omni.usd.get_context()

        # Skip if same file already open
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

    # =========================
    # UI Style
    # =========================
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
    # UI Build
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

                # -- Header --
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

                # -- Control Bar --
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

                # -- Mode Content Area (Window Swap) --
                with ui.ZStack():
                    # Container for LULA_IK mode
                    self.mode_ik_container = ui.VStack(visible=False, spacing=6)
                    with self.mode_ik_container:
                        with ui.ScrollingFrame(height=800):
                            with ui.VStack(spacing=8):
                                self._build_target_ui()
                    
                    # Container for JOINT mode
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

            # --- Left Target Card ---
            with ui.ZStack(height=240):
                ui.Rectangle(style={"background_color": 0x40151515, "border_radius": 10})
                with ui.VStack(spacing=2):
                    ui.Spacer(height=10)
                    # Title Row
                    with ui.HStack(height=24):
                        ui.Spacer(width=15)
                        ui.Label("Left Arm Target", style={"font_size": 15, "color": CLR_LEFT})
                        ui.Spacer(width=15)
                    
                    # Position Header & Checkbox
                    with ui.HStack(height=24):
                        ui.Spacer(width=15)
                        ui.Label("Position Targets", style=st["muted"], width=120)
                        ui.Spacer()
                        ui.Label("Orient", style=st["muted"], width=50)
                        self._cb_oriL = ui.CheckBox(width=20, style={"color": 0xFF000000})
                        self._cb_oriL.model.set_value(self._use_oriL)
                        self._cb_oriL.model.add_value_changed_fn(lambda m: self._on_toggle_ori("L", m.get_value_as_bool()))
                        ui.Spacer(width=15)

                    # Position Sliders
                    with ui.HStack(height=72): # 3 * 24
                        ui.Spacer(width=15)
                        with ui.VStack(spacing=0):
                            self._tgt_sliders_L["x"] = slider_row("X", -1.0, 1.0, self._tgtL["x"], lambda v: self._on_target_slider("L", "x", v), CLR_LEFT)
                            self._tgt_sliders_L["y"] = slider_row("Y", -1.0, 1.0, self._tgtL["y"], lambda v: self._on_target_slider("L", "y", v), CLR_LEFT)
                            self._tgt_sliders_L["z"] = slider_row("Z", 0.0, 1.5, self._tgtL["z"], lambda v: self._on_target_slider("L", "z", v), CLR_LEFT)
                        ui.Spacer(width=15)

                    ui.Spacer(height=6)

                    # Orientation Header
                    with ui.HStack(height=20):
                        ui.Spacer(width=15)
                        ui.Label("Orientation (RPY Degrees)", style=st["muted"])
                        ui.Spacer(width=15)

                    # Orientation Sliders
                    with ui.HStack(height=72): # 3 * 24
                        ui.Spacer(width=15)
                        with ui.VStack(spacing=0):
                            self._rpy_sliders_L["roll"] = slider_row("R", -180.0, 180.0, self._rpyL["roll"], lambda v: self._on_target_rpy_slider("L", "roll", v), CLR_LEFT)
                            self._rpy_sliders_L["pitch"] = slider_row("P", -90.0, 90.0, self._rpyL["pitch"], lambda v: self._on_target_rpy_slider("L", "pitch", v), CLR_LEFT)
                            self._rpy_sliders_L["yaw"] = slider_row("Y", -180.0, 180.0, self._rpyL["yaw"], lambda v: self._on_target_rpy_slider("L", "yaw", v), CLR_LEFT)
                        ui.Spacer(width=15)
                    ui.Spacer()

            # --- Right Target Card ---
            with ui.ZStack(height=240):
                ui.Rectangle(style={"background_color": 0x40151515, "border_radius": 10})
                with ui.VStack(spacing=2):
                    ui.Spacer(height=10)
                    # Title Row
                    with ui.HStack(height=24):
                        ui.Spacer(width=15)
                        ui.Label("Right Arm Target", style={"font_size": 15, "color": CLR_RIGHT})
                        ui.Spacer(width=15)
                    
                    # Position Header & Checkbox
                    with ui.HStack(height=24):
                        ui.Spacer(width=15)
                        ui.Label("Position Targets", style=st["muted"], width=120)
                        ui.Spacer()
                        ui.Label("Orient", style=st["muted"], width=50)
                        self._cb_oriR = ui.CheckBox(width=20, style={"color": 0xFF000000})
                        self._cb_oriR.model.set_value(self._use_oriR)
                        self._cb_oriR.model.add_value_changed_fn(lambda m: self._on_toggle_ori("R", m.get_value_as_bool()))
                        ui.Spacer(width=15)

                    # Position Sliders
                    with ui.HStack(height=72):
                        ui.Spacer(width=15)
                        with ui.VStack(spacing=0):
                            self._tgt_sliders_R["x"] = slider_row("X", -1.0, 1.0, self._tgtR["x"], lambda v: self._on_target_slider("R", "x", v), CLR_RIGHT)
                            self._tgt_sliders_R["y"] = slider_row("Y", -1.0, 1.0, self._tgtR["y"], lambda v: self._on_target_slider("R", "y", v), CLR_RIGHT)
                            self._tgt_sliders_R["z"] = slider_row("Z", 0.0, 1.5, self._tgtR["z"], lambda v: self._on_target_slider("R", "z", v), CLR_RIGHT)
                        ui.Spacer(width=15)

                    ui.Spacer(height=6)

                    # Orientation Header
                    with ui.HStack(height=20):
                        ui.Spacer(width=15)
                        ui.Label("Orientation (RPY Degrees)", style=st["muted"])
                        ui.Spacer(width=15)

                    # Orientation Sliders
                    with ui.HStack(height=72):
                        ui.Spacer(width=15)
                        with ui.VStack(spacing=0):
                            self._rpy_sliders_R["roll"] = slider_row("R", -180.0, 180.0, self._rpyR["roll"], lambda v: self._on_target_rpy_slider("R", "roll", v), CLR_RIGHT)
                            self._rpy_sliders_R["pitch"] = slider_row("P", -90.0, 90.0, self._rpyR["pitch"], lambda v: self._on_target_rpy_slider("R", "pitch", v), CLR_RIGHT)
                            self._rpy_sliders_R["yaw"] = slider_row("Y", -180.0, 180.0, self._rpyR["yaw"], lambda v: self._on_target_rpy_slider("R", "yaw", v), CLR_RIGHT)
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

        # Tab button highlight
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
        """Toggle UI visibility based on current mode"""
        is_joint = (self.mode == "JOINT")
        try:
            self.mode_joint_container.visible = is_joint
            self.mode_ik_container.visible = not is_joint
        except Exception:
            pass

        # Mode button color
        try:
            if is_joint:
                self.mode_btn.text = "Mode: JOINT"
                self.mode_btn.style = {"font_size": 13, "color": CLR_SUCCESS}
            else:
                self.mode_btn.text = "Mode: LULA_IK"
                self.mode_btn.style = {"font_size": 13, "color": CLR_HIGHLIGHT}
        except Exception:
            pass

    # ---------------- Mode ----------------
    def _toggle_mode(self):
        self.mode = "LULA_IK" if self.mode == "JOINT" else "JOINT"

        self._apply_mode_ui()

        stage = omni.usd.get_context().get_stage()
        if stage is not None:
            self._ensure_target_prims(stage)
            self._sync_targets_to_stage()

        self._set_status(f"Mode: {self.mode}")

    # ---------------- Update / Binding ----------------
    def _on_update(self, e):
        timeline = omni.timeline.get_timeline_interface()
        is_playing = timeline.is_playing()

        # If simulation STOPPED, perform a soft reset so we re-bind on next Play.
        # This prevents using stale physics handles.
        if not is_playing and self._bound:
            self._bound = False
            self._articulation_ready = False
            self._lula_ready = False
            self.articulation = None
            self.artik_left = None
            self.artik_right = None
            self._set_status("Simulation stopped. Ready to re-bind.")

        # Post-binding update loop
        if self._bound:
            stage = omni.usd.get_context().get_stage()
            if stage is not None:
                self._ensure_target_prims(stage)
                self._sync_targets_to_stage()

            # Apply IK action in LULA_IK mode
            if self.mode == "LULA_IK" and self._lula_ready and self._articulation_ready:
                try:
                    self._step_lula_ik(stage)
                except Exception:
                    if not self._printed_tb:
                        self._printed_tb = True
                        print("=== IK STEP TRACEBACK ===")
                        print(traceback.format_exc())

            # Apply dof targets in JOINT mode
            if self.mode == "JOINT":
                try:
                    self.dc.wake_up_articulation(self.art)
                except Exception:
                    pass

                for dof, tgt in self.targets.items():
                    try:
                        self.dc.set_dof_position_target(dof, float(tgt))
                    except Exception:
                        pass

            self._refresh_current_labels()
            return

        # Pre-binding: wait for Play
        if not is_playing:
            self._set_status("Press Play to start")
            return

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            self._set_status("Stage not ready...")
            return

        # 1) dynamic_control articulation bind
        art = self.dc.get_articulation(self.robot_root_prim)
        if art == 0:
            # Search under /World
            found_path, found_art = self._probe_articulation_under(stage, "/World")
            if found_art != 0:
                art = found_art

        if art == 0:
            self._set_status("Searching articulation...")
            return

        self._bind(art, self.robot_root_prim)

        # 2) SingleArticulation initialize (for IK)
        self._init_single_articulation(stage)

        # 3) Lula + ArticulationKinematicsSolver init
        self._init_lula_and_artik()

        # Create target prims
        self._ensure_target_prims(stage)
        self._sync_targets_to_stage()

    def _probe_articulation_under(self, stage, base_path: str):
        base = stage.GetPrimAtPath(base_path)
        if not base or not base.IsValid():
            return None, 0

        stack = [base]
        while stack:
            p = stack.pop()
            p_str = str(p.GetPath())

            if p_str.startswith("/World/Targets"):
                continue

            low = p_str.lower()
            if "constraint" in low or "actiongraph" in low:
                continue

            try:
                art = self.dc.get_articulation(p_str)
            except Exception:
                art = 0

            if art != 0:
                return p_str, art

            for c in p.GetChildren():
                stack.append(c)

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

    # ---------------- SingleArticulation ----------------
    def _init_single_articulation(self, stage):
        if self._articulation_ready and self.articulation is not None:
            return

        candidates = [
            self.robot_root_prim,
            f"{self.robot_root_prim}/root_joint",
            f"{self.robot_root_prim}/openarm_left_ee_tcp",
            f"{self.robot_root_prim}/openarm_right_ee_tcp",
        ]

        for prim_path in candidates:
            prim = stage.GetPrimAtPath(prim_path)
            if not prim or not prim.IsValid():
                continue

            try:
                art = SingleArticulation(prim_path)
                if not art.handles_initialized:
                    art.initialize()

                self.articulation = art
                self._articulation_ready = True
                self._set_status(f"Articulation ready: {prim_path}")
                return
            except Exception:
                continue

        self._set_status("SingleArticulation init failed")

    # ---------------- Lula + ArticulationKinematicsSolver ----------------
    def _init_lula_and_artik(self):
        if self._lula_ready or self._lula_failed:
            return

        if not self._articulation_ready or self.articulation is None:
            return

        ext_mgr = omni.kit.app.get_app().get_extension_manager()
        ext_root = ext_mgr.get_extension_path(self._ext_id)
        assets = os.path.join(ext_root, "openarm_UI", "assets")

        urdf_path = os.path.join(assets, "openarm_bimanual.urdf")
        left_yaml = os.path.join(assets, "openarm_left.yaml")
        right_yaml = os.path.join(assets, "openarm_right.yaml")

        for p in [urdf_path, left_yaml, right_yaml]:
            if not os.path.exists(p):
                self._lula_failed = True
                self._set_status(f"Lula file missing: {p}")
                return

        try:
            self.lula_left = LulaKinematicsSolver(left_yaml, urdf_path)
            self.lula_right = LulaKinematicsSolver(right_yaml, urdf_path)

            self.artik_left = ArticulationKinematicsSolver(self.articulation, self.lula_left, self.ee_left)
            self.artik_right = ArticulationKinematicsSolver(self.articulation, self.lula_right, self.ee_right)

            self._lula_ready = True
            self._set_status("Lula IK ready")
            
            # Auto-initialize targets to current EE pose
            self._set_target_to_current("L")
            self._set_target_to_current("R")
        except Exception as e:
            self._lula_failed = True
            self._set_status(f"Lula/ArtIK init failed: {e}")
            print("[OpenArmAutoController] Lula/ArtIK init failed:\n", traceback.format_exc())

    # ---------------- Target prim (Cube) ----------------
    def _ensure_target_prims(self, stage):
        targets_root = "/World/Targets"
        if not stage.GetPrimAtPath(targets_root).IsValid():
            stage.DefinePrim(targets_root, "Xform")

        def ensure_target(path_xform: str, default_translate: Gf.Vec3d, default_quat_wxyz):
            prim = stage.GetPrimAtPath(path_xform)
            if not prim or not prim.IsValid():
                prim = stage.DefinePrim(path_xform, "Xform")

            xf = UsdGeom.Xformable(prim)

            # Translate op
            t_op = None
            for op in xf.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    t_op = op
                    break
            if t_op is None:
                t_op = xf.AddTranslateOp()
            if not t_op.GetAttr().HasAuthoredValueOpinion():
                t_op.Set(default_translate)

            # Orient op
            o_op = None
            for op in xf.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                    o_op = op
                    break
            if o_op is None:
                o_op = xf.AddOrientOp()
            w, x, y, z = default_quat_wxyz
            q = Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z)))
            if not o_op.GetAttr().HasAuthoredValueOpinion():
                o_op.Set(q)

            # Cube visual
            vis_path = f"{path_xform}/vis"
            vis_prim = stage.GetPrimAtPath(vis_path)
            if not vis_prim or not vis_prim.IsValid():
                vis_prim = stage.DefinePrim(vis_path, "Cube")
                cube = UsdGeom.Cube(vis_prim)
                cube.CreateSizeAttr(0.04)
                
            # Axis markers (X:Red, Y:Green, Z:Blue)
            def ensure_axis(axis_name, color, rotation_quat=None):
                axis_path = f"{path_xform}/{axis_name}"
                prim = stage.GetPrimAtPath(axis_path)
                if not prim or not prim.IsValid():
                    prim = stage.DefinePrim(axis_path, "Cylinder")
                    cyl = UsdGeom.Cylinder(prim)
                    cyl.CreateRadiusAttr(0.002)
                    cyl.CreateHeightAttr(0.12)
                    cyl.CreateDisplayColorAttr([color])
                    xf = UsdGeom.Xformable(prim)
                    # Position at the end of the cylinder to point outwards
                    t_op = xf.AddTranslateOp()
                    if axis_name == "X":
                        t_op.Set(Gf.Vec3d(0.06, 0, 0))
                        r_op = xf.AddRotateYOp()
                        r_op.Set(90)
                    elif axis_name == "Y":
                        t_op.Set(Gf.Vec3d(0, 0.06, 0))
                        r_op = xf.AddRotateXOp()
                        r_op.Set(-90)
                    elif axis_name == "Z":
                        t_op.Set(Gf.Vec3d(0, 0, 0.06))

            ensure_axis("X", (1, 0, 0))
            ensure_axis("Y", (0, 1, 0))
            ensure_axis("Z", (0, 0, 1))

        qL = rpy_to_quat_wxyz(self._rpyL["roll"], self._rpyL["pitch"], self._rpyL["yaw"])
        qR = rpy_to_quat_wxyz(self._rpyR["roll"], self._rpyR["pitch"], self._rpyR["yaw"])
        ensure_target(self.left_target_prim,
                      Gf.Vec3d(self._tgtL["x"], self._tgtL["y"], self._tgtL["z"]),
                      (float(qL[0]), float(qL[1]), float(qL[2]), float(qL[3])))
        ensure_target(self.right_target_prim,
                      Gf.Vec3d(self._tgtR["x"], self._tgtR["y"], self._tgtR["z"]),
                      (float(qR[0]), float(qR[1]), float(qR[2]), float(qR[3])))

    def _get_or_create_translate_op(self, stage, prim_path: str):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return None
        x = UsdGeom.Xformable(prim)
        for op in x.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                return op
        return x.AddTranslateOp()

    def _set_target_translation(self, stage, prim_path: str, xyz):
        op = self._get_or_create_translate_op(stage, prim_path)
        if op is None:
            return
        op.Set(Gf.Vec3d(float(xyz[0]), float(xyz[1]), float(xyz[2])))

    def _get_or_create_orient_op(self, stage, prim_path: str):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return None
        xf = UsdGeom.Xformable(prim)
        for op in xf.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                return op
        return xf.AddOrientOp()

    def _set_target_orientation_wxyz(self, stage, prim_path: str, quat_wxyz):
        op = self._get_or_create_orient_op(stage, prim_path)
        if op is None:
            return
        w, x, y, z = quat_wxyz
        q = Gf.Quatf(float(w), Gf.Vec3f(float(x), float(y), float(z)))
        op.Set(q)

    def _get_world_translation(self, stage, prim_path: str):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return None
        xf = UsdGeom.Xformable(prim)
        M = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        t = M.ExtractTranslation()
        return (float(t[0]), float(t[1]), float(t[2]))

    def _get_world_quat_wxyz(self, stage, prim_path: str):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            return None
        xf = UsdGeom.Xformable(prim)
        M = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        r = M.ExtractRotationQuat()
        w = float(r.GetReal())
        im = r.GetImaginary()
        return (w, float(im[0]), float(im[1]), float(im[2]))

    def _on_target_slider(self, side: str, axis: str, value: float):
        if self._is_syncing:
            return
        if side == "L":
            self._tgtL[axis] = float(value)
        else:
            self._tgtR[axis] = float(value)

        stage = omni.usd.get_context().get_stage()
        if stage is not None:
            self._ensure_target_prims(stage)
            self._sync_targets_to_stage()

    def _on_target_rpy_slider(self, side: str, axis: str, value: float):
        if self._is_syncing:
            return
        if side == "L":
            self._rpyL[axis] = float(value)
        else:
            self._rpyR[axis] = float(value)

        stage = omni.usd.get_context().get_stage()
        if stage is not None:
            self._ensure_target_prims(stage)
            self._sync_targets_to_stage()

    def _on_toggle_ori(self, side: str, value: bool):
        if side == "L":
            self._use_oriL = value
        else:
            self._use_oriR = value
        self._set_status(f"{side} orientation constraint: {value}")

    def _reset_rpy(self, side: str):
        if side == "L":
            self._rpyL = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
        else:
            self._rpyR = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
        self._sync_targets_to_stage()
        self._set_status(f"{side} orientation reset")

    def _reset_all_rpy(self):
        self._rpyL = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
        self._rpyR = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
        stage = omni.usd.get_context().get_stage()
        if stage is not None:
            self._ensure_target_prims(stage)
            self._sync_targets_to_stage()
        self._set_status("All orientations reset")

    def _sync_targets_to_stage(self):
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        self._ensure_target_prims(stage)
        self._set_target_translation(stage, self.left_target_prim, (self._tgtL["x"], self._tgtL["y"], self._tgtL["z"]))
        self._set_target_translation(stage, self.right_target_prim, (self._tgtR["x"], self._tgtR["y"], self._tgtR["z"]))

        qL = rpy_to_quat_wxyz(self._rpyL["roll"], self._rpyL["pitch"], self._rpyL["yaw"])
        qR = rpy_to_quat_wxyz(self._rpyR["roll"], self._rpyR["pitch"], self._rpyR["yaw"])
        self._set_target_orientation_wxyz(stage, self.left_target_prim, qL)
        self._set_target_orientation_wxyz(stage, self.right_target_prim, qR)
        
        # Also sync UI sliders
        self._sync_ui_from_state()

    def _sync_ui_from_state(self):
        """Force UI sliders to match internal _tgt and _rpy state"""
        if self._is_syncing:
            return
        self._is_syncing = True
        try:
            # Left
            for k in ["x", "y", "z"]:
                if k in self._tgt_sliders_L: self._tgt_sliders_L[k].model.set_value(self._tgtL[k])
            for k in ["roll", "pitch", "yaw"]:
                if k in self._rpy_sliders_L: self._rpy_sliders_L[k].model.set_value(self._rpyL[k])
            # Right
            for k in ["x", "y", "z"]:
                if k in self._tgt_sliders_R: self._tgt_sliders_R[k].model.set_value(self._tgtR[k])
            for k in ["roll", "pitch", "yaw"]:
                if k in self._rpy_sliders_R: self._rpy_sliders_R[k].model.set_value(self._rpyR[k])
        finally:
            self._is_syncing = False

    def _reset_targets(self):
        # Reset to initial default positions
        self._tgtL = {"x": 0.0, "y": 0.1535, "z": 0.0689}
        self._tgtR = {"x": 0.0, "y": -0.1535, "z": 0.0689}
        self._rpyL = {"roll": 180.0, "pitch": 0.0, "yaw": 0.0}
        self._rpyR = {"roll": 180.0, "pitch": 0.0, "yaw": 0.0}
        
        self._sync_targets_to_stage()
        self._set_status("Targets reset to initial positions")

    def _reset_to_current_all(self):
        self._set_target_to_current("L")
        self._set_target_to_current("R")
        self._sync_targets_to_stage()
        self._set_status("Targets set to current robot pose")

    def _set_target_to_current(self, side: str):
        """Set target position/orientation to the robot's current end-effector pose"""
        if self.articulation is None or not self.articulation.handles_initialized:
            return
        if self.articulation.get_joint_positions() is None:
            return
        
        ik_solver = self.artik_left if side == "L" else self.artik_right
        if ik_solver is None:
            return
            
        try:
            # Get current EE pose (position, 3x3 rotation matrix)
            pos, rot_mat = ik_solver.compute_end_effector_pose()
            
            # Convert rot_mat to Euler angles in degrees (extrinsic XYZ)
            euler = matrix_to_euler_angles(rot_mat, degrees=True)
            
            # Euler normalization: if roll is ~180, it might be an inverted solution.
            # We can try to normalize it to [0, 180] or similar if needed, 
            # but for now let's just ensure it's precisely handled.
            r, p, y = float(euler[0]), float(euler[1]), float(euler[2])
            
            # Simple heuristic: if |roll| > 170 and |pitch| < 10, it's often a flipped representation of (0, 0, 180) or similar.
            # But the most robust way is to just use what Isaac Sim gives us and SYNC the UI.
            
            if side == "L":
                self._tgtL = {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])}
                self._rpyL = {"roll": r, "pitch": p, "yaw": y}
                # Sync sliders
                for k in ["x", "y", "z"]:
                    if k in self._tgt_sliders_L: self._tgt_sliders_L[k].model.set_value(self._tgtL[k])
                for k in ["roll", "pitch", "yaw"]:
                    if k in self._rpy_sliders_L: self._rpy_sliders_L[k].model.set_value(self._rpyL[k])
            else:
                self._tgtR = {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])}
                self._rpyR = {"roll": r, "pitch": p, "yaw": y}
                # Sync sliders
                for k in ["x", "y", "z"]:
                    if k in self._tgt_sliders_R: self._tgt_sliders_R[k].model.set_value(self._tgtR[k])
                for k in ["roll", "pitch", "yaw"]:
                    if k in self._rpy_sliders_R: self._rpy_sliders_R[k].model.set_value(self._rpyR[k])
                
            self._sync_targets_to_stage()
            self._set_status(f"{side} Target initialized from current pose")
        except Exception as e:
            self._set_status(f"Error setting {side} target: {e}")

    # ---------------- IK Step ----------------
    def _step_lula_ik(self, stage):
        if stage is None or self.articulation is None:
            return
        if not self.articulation.handles_initialized:
            return
        if self.articulation.get_joint_positions() is None:
            return
            
        if self.artik_left is None or self.artik_right is None:
            return

        # 1) Sync robot base pose to solvers (Crucial if robot moved from origin)
        pos_root, quat_root = self.articulation.get_world_pose()
        self.lula_left.set_robot_base_pose(pos_root, quat_root)
        self.lula_right.set_robot_base_pose(pos_root, quat_root)

        # 2) Get targets
        tgtL = self._get_world_translation(stage, self.left_target_prim)
        tgtR = self._get_world_translation(stage, self.right_target_prim)
        qL_wxyz = self._get_world_quat_wxyz(stage, self.left_target_prim)
        qR_wxyz = self._get_world_quat_wxyz(stage, self.right_target_prim)

        if tgtL is None or tgtR is None:
            return

        posL = np.array(tgtL, dtype=np.float32)
        posR = np.array(tgtR, dtype=np.float32)

        # Use orientation from RPY if available AND enabled, else position-only
        oriL = None
        oriR = None
        
        if self._use_oriL and qL_wxyz is not None:
            oriL = quat_normalize_wxyz(qL_wxyz).astype(np.float32)
        if self._use_oriR and qR_wxyz is not None:
            oriR = quat_normalize_wxyz(qR_wxyz).astype(np.float32)

        # 3) Compute IK for both arms
        actionL, succL = self.artik_left.compute_inverse_kinematics(posL, oriL)
        actionR, succR = self.artik_right.compute_inverse_kinematics(posR, oriR)

        # 4) Merge and Apply actions in one go to avoid overwriting
        combined_positions = []
        combined_indices = []
        
        if succL:
            combined_positions.extend(actionL.joint_positions)
            combined_indices.extend(actionL.joint_indices)
        if succR:
            combined_positions.extend(actionR.joint_positions)
            combined_indices.extend(actionR.joint_indices)

        if combined_indices:
            merged_action = ArticulationAction(
                joint_positions=np.array(combined_positions),
                joint_indices=np.array(combined_indices)
            )
            self.articulation.get_articulation_controller().apply_action(merged_action)

        # 5) Status feedback
        self._set_status(f"IK Status -> Left:{'OK' if succL else 'FAIL'} Right:{'OK' if succR else 'FAIL'}")

    # ---------------- Slider build ----------------
    def _build_sliders(self):
        dof_count = self.dc.get_articulation_dof_count(self.art)
        name_map = {}

        for i in range(dof_count):
            dof = self.dc.get_articulation_dof(self.art, i)
            name = self.dc.get_dof_name(dof)
            try:
                cur = float(self.dc.get_dof_position(dof))
            except Exception:
                cur = 0.0
            name_map[name] = (dof, cur)

        st = self._style()

        with self.left_container:
            self._section_header("Left Arm", CLR_LEFT)
            for name in [n for n in VISIBLE_DOFS if is_left(n)]:
                self._add_joint_row(name, name_map, st)

        with self.right_container:
            self._section_header("Right Arm", CLR_RIGHT)
            for name in [n for n in VISIBLE_DOFS if is_right(n)]:
                self._add_joint_row(name, name_map, st)

    def _add_joint_row(self, name: str, name_map: dict, st: dict):
        if name not in name_map:
            ui.Label(f"missing: {name}", style=st["muted"])
            return

        dof, cur_rad = name_map[name]
        grip = (USE_GRIPPER_RAW and is_gripper(name))

        if grip:
            lo, hi = GRIPPER_RAW_LIMIT.get(name, (0.0, 0.04))
            cur_ui = float(cur_rad)
            unit = ""
            tgt_ui = cur_ui
            self.targets[dof] = float(cur_rad)
        else:
            lo, hi = JOINT_LIMIT_DEG.get(name, (-180.0, 180.0))
            cur_ui = math.degrees(float(cur_rad))
            unit = "deg"
            tgt_ui = cur_ui
            self.targets[dof] = float(cur_rad)

        # Short joint name display (openarm_left_joint3 -> J3)
        short = name.split("_")[-1].upper()
        if "finger" in name.lower():
            short = "GRIP"

        with ui.VStack(spacing=2):
            with ui.HStack(height=22):
                ui.Label(f"{short}", style={"font_size": 14, "color": CLR_TEXT}, width=50)
                lbl = ui.Label(
                    f"cur:{fmt1(cur_ui)}{unit}  tgt:{fmt1(tgt_ui)}{unit}",
                    style=st["curtgt"]
                )
                self.cur_tgt_labels[dof] = lbl

            s = ui.FloatSlider(min=float(lo), max=float(hi), style={"font_size": 12})
            s.model.set_value(float(tgt_ui))

            self.slider_by_dof[dof] = s
            self.dof_name_by_handle[dof] = name

            s.model.add_value_changed_fn(
                lambda m, n=name, h=dof, slider=s: self._on_joint_slider(n, h, slider)
            )

            with ui.HStack(height=16):
                ui.Label(f"{fmt1(lo)}{unit}", style=st["range"], width=70)
                ui.Spacer()
                ui.Label(f"{fmt1(hi)}{unit}", style=st["range"], width=70,
                         alignment=ui.Alignment.RIGHT_CENTER)

    def _refresh_current_labels(self):
        for dof, slider in self.slider_by_dof.items():
            name = self.dof_name_by_handle.get(dof, "")
            grip = (USE_GRIPPER_RAW and is_gripper(name))

            try:
                cur = float(self.dc.get_dof_position(dof))
            except Exception:
                continue

            try:
                tgt_ui = float(slider.model.get_value_as_float())
            except Exception:
                tgt_ui = 0.0

            if grip:
                cur_ui = cur
                unit = ""
            else:
                cur_ui = math.degrees(cur)
                unit = "deg"

            if dof in self.cur_tgt_labels:
                self.cur_tgt_labels[dof].text = f"cur:{fmt1(cur_ui)}{unit}  tgt:{fmt1(tgt_ui)}{unit}"

    def _on_joint_slider(self, dof_name: str, dof_handle: int, slider):
        if not self._bound:
            return

        # Prevent slider override in IK mode
        if self.mode == "LULA_IK":
            return

        val_ui = float(slider.model.get_value_as_float())
        grip = (USE_GRIPPER_RAW and is_gripper(dof_name))

        if grip:
            self.targets[dof_handle] = val_ui
        else:
            self.targets[dof_handle] = math.radians(val_ui)

    def _on_zero_position(self):
        if not self._bound:
            self._set_status("Not bound yet (press Play)")
            return

        # Reset sliders to zero
        for dof, slider in self.slider_by_dof.items():
            slider.model.set_value(0.0)

        # Reset JOINT mode targets to zero
        for dof in list(self.targets.keys()):
            self.targets[dof] = 0.0

        self._set_status("Zero Position applied")
        self._refresh_current_labels()

    def on_shutdown(self):
        self._sub = None
        if getattr(self, "window", None) is not None:
            try:
                self.window.destroy()
            except Exception:
                pass
            self.window = None
        self._set_status("shutdown")