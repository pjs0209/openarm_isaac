import omni.ext
import omni.ui as ui
import omni.timeline
import omni.kit.app
import omni.usd
import math
import numpy as np
import os

from pxr import UsdGeom, Gf, Usd
from omni.isaac.dynamic_control import _dynamic_control


# =========================
# 설정
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

# degree 범위 (팔 관절)
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

# 그리퍼 raw(0~0.05)로 운용
USE_GRIPPER_RAW = True
GRIPPER_RAW_LIMIT = {
    "openarm_left_finger_joint1": (0.0, 0.04),
    "openarm_right_finger_joint1": (0.0, 0.04),
}


def fmt1(x: float) -> str:
    """소수점 2자리 고정"""
    return f"{x:.2f}"


def is_left(name: str) -> bool:
    return "_left_" in name


def is_right(name: str) -> bool:
    return "_right_" in name


def is_gripper(name: str) -> bool:
    return "finger" in name.lower()


class OpenArmAutoController(omni.ext.IExt):
    def on_startup(self, ext_id):
        # ✅ 핫리로드/Enable 반복 시 창이 중복 생성되는 것 방지
        if getattr(self, "window", None) is not None:
            try:
                self.window.destroy()
            except Exception:
                pass
            self.window = None

        # =========================
        # ✅ 여기서 USD를 "덮어씌우기(open_stage)"로 자동 로드
        # =========================
        self._auto_open_stage(ext_id)

        # -------------------------
        # 기존 초기화
        # -------------------------
        self.preferred_root_path = "/World/openarm"
        self.base_search_path = "/World"

        self.dc = _dynamic_control.acquire_dynamic_control_interface()

        self._bound = False
        self.art = 0

        # dof_handle -> target (rad or raw)
        self.targets = {}

        # dof_handle -> ui.FloatSlider
        self.slider_by_dof = {}

        # dof_handle -> dof_name
        self.dof_name_by_handle = {}

        # dof_handle -> ui.Label  (현재/목표 표시 라벨)
        self.cur_tgt_labels = {}

        # 탭 상태
        self._active_tab = "left"

        self._build_ui()

        app = omni.kit.app.get_app()
        self._sub = app.get_update_event_stream().create_subscription_to_pop(self._on_update)

        self._set_status("Press Play Button")

    # =========================
    # ✅ 덮어씌우기 로더
    # =========================
    def _auto_open_stage(self, ext_id: str):
    
        ext_mgr = omni.kit.app.get_app().get_extension_manager()
        ext_root = ext_mgr.get_extension_path(ext_id) 
        usd_path = os.path.join(ext_root, "openarm_UI", "assets", "usd", "openarm_sim.usd")
        usd_path = usd_path.replace("\\", "/")

        if not os.path.exists(usd_path):
            self._set_status(f"USD not found: {usd_path}")
            return

        ctx = omni.usd.get_context()

        # ✅ 같은 파일이면 재오픈 방지(Enable/핫리로드 대비)
        try:
            cur_url = ctx.get_stage_url()
            if cur_url and (cur_url.endswith("openarm_sim.usd")):
                print("[OpenArmAutoController] stage already opened:", cur_url)
                return
        except Exception:
            pass

        print("[OpenArmAutoController] open_stage:", usd_path)
        ctx.open_stage(usd_path)

        # open_stage는 비동기일 수 있으니 바인딩은 _on_update에서 계속 시도하도록 둠
        self._bound = False
        self.art = 0

    # ---------------- UI helpers ----------------
    def _style(self):
        return {
            "title": {"font_size": 24},
            "muted": {"font_size": 18, "color": 0xFFB0B0B0},
            "section": {"font_size": 18},
            "joint_name": {"font_size": 20},
            "range": {"font_size": 18},
            "curtgt": {"font_size": 20, "color": 0xFFFFFFFF},
        }

    def _build_ui(self):
        st = self._style()

        self.window = ui.Window(
            "OpenArm Controller",
            width=520,
            height=900,
            dockPreference=ui.DockPreference.LEFT
        )

        with self.window.frame:
            with ui.VStack(spacing=8):
                # Header
                with ui.VStack(height=100):
                    ui.Label("OpenArm Controller", style=st["title"])
                    self.status = ui.Label("", style=st["muted"])

                # 버튼 줄 (Zero + Tab buttons)
                with ui.HStack(height=36, spacing=8):
                    ui.Button("Zero Position", clicked_fn=self._on_zero_position)
                    ui.Spacer()
                    self.left_tab_btn = ui.Button("Left Arm", clicked_fn=lambda: self._show_tab("left"))
                    self.right_tab_btn = ui.Button("Right Arm", clicked_fn=lambda: self._show_tab("right"))

                ui.Separator(height=2)

                # Content
                with ui.ScrollingFrame():
                    with ui.VStack(height=700, spacing=12):
                        self.left_container = ui.VStack(spacing=8)
                        self.right_container = ui.VStack(spacing=8)

        self._show_tab(self._active_tab)

    def _section_header(self, title: str):
        st = self._style()
        with ui.HStack(height=26):
            ui.Label(title, style=st["section"])
        ui.Separator(height=2)

    def _show_tab(self, name: str):
        self._active_tab = name
        if getattr(self, "left_container", None) is None or getattr(self, "right_container", None) is None:
            return
        try:
            self.left_container.visible = (name == "left")
            self.right_container.visible = (name == "right")
        except Exception:
            pass

    def _set_status(self, s: str):
        if getattr(self, "_last_status", None) == s:
            return
        self._last_status = s
        if getattr(self, "status", None) is not None:
            self.status.text = s
        print("[OpenArmAutoController]", s)

    # ----------------  Binding / Update ----------------
    def _on_update(self, e):
        if self._bound:
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

        timeline = omni.timeline.get_timeline_interface()
        if not timeline.is_playing():
            self._set_status("Press Play Button")
            return

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            self._set_status("stage not ready")
            return

        art = self.dc.get_articulation(self.preferred_root_path)
        if art != 0:
            self._bind(art, self.preferred_root_path)
            return

        found_path, found_art = self._probe_articulation_under(stage, self.base_search_path)
        if found_art != 0:
            self._bind(found_art, found_path)
            return

        self._set_status("searching articulation...")

    def _bind(self, art_handle: int, root_path: str):
        self.art = art_handle
        self._bound = True

        self.targets.clear()
        self.slider_by_dof.clear()
        self.dof_name_by_handle.clear()
        self.cur_tgt_labels.clear()

        self._set_status(f"complete bounding: {root_path}")
        self._build_sliders()
        self._show_tab(self._active_tab)

    def _probe_articulation_under(self, stage, base_path: str):
        base = stage.GetPrimAtPath(base_path)
        if not base or not base.IsValid():
            return None, 0

        stack = [base]
        while stack:
            p = stack.pop()
            p_str = str(p.GetPath())
            low = p_str.lower()

            if "constraint" in low:
                continue

            if "actiongraph" in low:
                continue

            art = self.dc.get_articulation(p_str)
            if art != 0:
                return p_str, art

            for c in p.GetChildren():
                stack.append(c)

        return None, 0

    # ----------------  Slider build ----------------
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
            self._section_header("Left Arm")
            for name in [n for n in VISIBLE_DOFS if is_left(n)]:
                self._add_joint_row(name, name_map, st)

        with self.right_container:
            self._section_header("Right Arm")
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
            unit = "°"
            tgt_ui = cur_ui
            self.targets[dof] = float(cur_rad)

        with ui.VStack(spacing=4):
            with ui.HStack(height=26):
                ui.Label(name, style=st["joint_name"], width=260)
                lbl = ui.Label(
                    f"cur: {fmt1(cur_ui)}{unit}   tar: {fmt1(tgt_ui)}{unit}",
                    style=st["curtgt"]
                )
                self.cur_tgt_labels[dof] = lbl

            s = ui.FloatSlider(min=float(lo), max=float(hi))
            s.model.set_value(float(tgt_ui))

            self.slider_by_dof[dof] = s
            self.dof_name_by_handle[dof] = name

            s.model.add_value_changed_fn(
                lambda m, n=name, h=dof, slider=s: self._on_slider(n, h, slider)
            )

            with ui.HStack(height=22):
                ui.Label(f"{fmt1(lo)}{unit}", style=st["range"], width=110)
                ui.Spacer()
                ui.Label(
                    f"{fmt1(hi)}{unit}",
                    style=st["range"],
                    width=110,
                    alignment=ui.Alignment.RIGHT_CENTER
                )

            ui.Separator(height=1)

    # ----------------  Update label + callbacks ----------------
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
                unit = "°"

            if dof in self.cur_tgt_labels:
                self.cur_tgt_labels[dof].text = f"cur: {fmt1(cur_ui)}{unit}   tar: {fmt1(tgt_ui)}{unit}"

    def _on_slider(self, dof_name: str, dof_handle: int, slider):
        if not self._bound:
            return

        val_ui = float(slider.model.get_value_as_float())
        grip = (USE_GRIPPER_RAW and is_gripper(dof_name))

        if grip:
            self.targets[dof_handle] = val_ui
        else:
            self.targets[dof_handle] = math.radians(val_ui)

    # ----------------  Zero Position ----------------
    def _on_zero_position(self):
        if not self._bound:
            self._set_status("It's not completely finished yet. Please try again later.")
            return

        for dof, slider in self.slider_by_dof.items():
            name = self.dof_name_by_handle.get(dof, "")
            grip = (USE_GRIPPER_RAW and is_gripper(name))

            slider.model.set_value(0.0)

            if grip:
                self.targets[dof] = 0.0
            else:
                self.targets[dof] = 0.0

        self._set_status("Zero Position applied.")
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