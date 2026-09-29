"""Small URDF/FK dashboard for human takeover synchronization.

Real D1 mesh views with orthographic cameras and a fixed dorsal hand view.
Forward kinematics only: no physics steps or robot commands.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from dataclasses import dataclass
import os
from pathlib import Path
from xml.etree import ElementTree

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np


ARM_JOINTS = tuple(f"joint_{index}" for index in range(1, 7))
HAND_JOINTS = (
    "thumb_cmc_yaw",
    "thumb_cmc_pitch",
    "thumb_ip",
    "index_mcp_pitch",
    "index_dip",
    "middle_mcp_pitch",
    "middle_dip",
    "ring_mcp_pitch",
    "ring_dip",
    "pinky_mcp_pitch",
    "pinky_dip",
)
# The hardware exposes six hand channels while the URDF has coupled finger joints.
HAND_CHANNELS = (1, 0, 0, 2, 2, 3, 3, 4, 4, 5, 5)
ARM_DIRECTION_KINDS = (
    "horizontal",
    "vertical",
    "vertical",
    "rotation",
    "vertical",
    "rotation",
)
ARM_DIRECTION_NAMES = (
    ("LEFT", "RIGHT"),
    ("DOWN", "UP"),
    ("DOWN", "UP"),
    ("CCW", "CW"),
    ("DOWN", "UP"),
    ("CCW", "CW"),
)
HAND_DIRECTION_KINDS = ("vertical", "rotation", "vertical", "vertical", "vertical", "vertical")
HAND_DIRECTION_NAMES = (
    ("CLOSE", "OPEN"),
    ("CCW", "CW"),
    ("CLOSE", "OPEN"),
    ("CLOSE", "OPEN"),
    ("CLOSE", "OPEN"),
    ("CLOSE", "OPEN"),
)


def _x11_focus_window() -> int | None:
    library = ctypes.util.find_library("X11")
    if not library:
        return None
    try:
        x11 = ctypes.CDLL(library)
        x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
        x11.XOpenDisplay.restype = ctypes.c_void_p
        x11.XGetInputFocus.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_ulong),
            ctypes.POINTER(ctypes.c_int),
        ]
        x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
        display = x11.XOpenDisplay(None)
        if not display:
            return None
        window, revert = ctypes.c_ulong(), ctypes.c_int()
        x11.XGetInputFocus(display, ctypes.byref(window), ctypes.byref(revert))
        x11.XCloseDisplay(display)
        return window.value if window.value > 1 else None
    except (AttributeError, OSError):
        return None


def _restore_x11_focus(window: int | None) -> None:
    if not window:
        return
    library = ctypes.util.find_library("X11")
    if not library:
        return
    try:
        x11 = ctypes.CDLL(library)
        x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
        x11.XOpenDisplay.restype = ctypes.c_void_p
        x11.XSetInputFocus.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_int,
            ctypes.c_ulong,
        ]
        x11.XFlush.argtypes = [ctypes.c_void_p]
        x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
        display = x11.XOpenDisplay(None)
        if not display:
            return
        x11.XSetInputFocus(display, ctypes.c_ulong(window), 0, 0)
        x11.XFlush(display)
        x11.XCloseDisplay(display)
    except (AttributeError, OSError):
        pass


@dataclass(frozen=True)
class _Joint:
    name: str
    kind: str
    parent: str
    child: str
    lower: float
    upper: float


class UrdfKinematics:
    """Pinocchio-backed FK adapter for the dashboard's 12 control channels."""

    def __init__(self, urdf_path: str | Path):
        try:
            import pinocchio as pin
        except ImportError as exc:
            raise RuntimeError(
                "Human sync FK needs pinocchio; install pinocchio in the runtime venv"
            ) from exc

        self.pin = pin
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()
        root = ElementTree.parse(urdf_path).getroot()
        self.joints: dict[str, _Joint] = {}
        self.children: dict[str, list[_Joint]] = {}
        links = {element.attrib["name"] for element in root.findall("link")}
        for element in root.findall("joint"):
            name = element.attrib["name"]
            parent = element.find("parent")
            child = element.find("child")
            if parent is None or child is None:
                raise ValueError(f"URDF joint {name!r} must have parent and child")
            limit = element.find("limit")
            lower = float(limit.attrib.get("lower", "-inf")) if limit is not None else -np.inf
            upper = float(limit.attrib.get("upper", "inf")) if limit is not None else np.inf
            joint = _Joint(
                name=name,
                kind=element.attrib.get("type", "fixed"),
                parent=parent.attrib["link"],
                child=child.attrib["link"],
                lower=lower,
                upper=upper,
            )
            if name in self.joints:
                raise ValueError(f"duplicate URDF joint {name!r}")
            self.joints[name] = joint
            self.children.setdefault(joint.parent, []).append(joint)

        if not self.joints:
            raise ValueError(f"URDF has no joints: {urdf_path}")
        child_links = {joint.child for joint in self.joints.values()}
        roots = links - child_links
        if len(roots) != 1:
            raise ValueError(f"URDF must have one root link, got {sorted(roots)}")
        self.root_link = roots.pop()
        self.link_names = links
        self.child_joint = {joint.child: joint for joint in self.joints.values()}
        missing = set(ARM_JOINTS + HAND_JOINTS) - self.joints.keys()
        if missing:
            raise ValueError(f"URDF is missing dashboard joints: {sorted(missing)}")
        missing_model_joints = [
            name for name in ARM_JOINTS + HAND_JOINTS
            if not self.model.existJointName(name)
        ]
        if missing_model_joints:
            raise ValueError(
                f"Pinocchio model is missing dashboard joints: {sorted(missing_model_joints)}"
            )
        self._joint_ids = {
            name: self.model.getJointId(name)
            for name in ARM_JOINTS + HAND_JOINTS
        }
        self.arm_edges = self._path_edges(ARM_JOINTS)
        hand_root = self.joints[ARM_JOINTS[-1]].child
        self.hand_edges = self._descendant_edges(hand_root)
        self.hand_root = hand_root

    def _path_edges(self, names: tuple[str, ...]) -> tuple[str, ...]:
        selected: set[str] = set()
        for name in names:
            joint = self.joints[name]
            while True:
                selected.add(joint.name)
                parent_joint = self.child_joint.get(joint.parent)
                if parent_joint is None:
                    break
                joint = parent_joint
        return tuple(name for name in self.joints if name in selected)

    def _descendant_edges(self, root_link: str) -> tuple[str, ...]:
        selected: list[str] = []
        stack = [root_link]
        while stack:
            parent = stack.pop()
            for joint in self.children.get(parent, ()):
                selected.append(joint.name)
                stack.append(joint.child)
        return tuple(selected)

    def pose(self, state: np.ndarray) -> dict[str, object]:
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape != (12,) or not np.isfinite(state).all():
            raise ValueError(f"dashboard state must be finite shape (12,), got {state.shape}")
        q = np.zeros(self.model.nq, dtype=np.float64)
        for index, name in enumerate(ARM_JOINTS):
            q[self.model.idx_qs[self._joint_ids[name]]] = float(state[index])
        for name, channel in zip(HAND_JOINTS, HAND_CHANNELS):
            normalized = float(np.clip(state[6 + channel], 0.0, 1.0))
            joint_id = self._joint_ids[name]
            q_index = self.model.idx_qs[joint_id]
            lower = float(self.model.lowerPositionLimit[q_index])
            upper = float(self.model.upperPositionLimit[q_index])
            if np.isfinite(lower) and np.isfinite(upper):
                q[q_index] = lower + normalized * (upper - lower)
            else:
                q[q_index] = normalized

        self.pin.framesForwardKinematics(self.model, self.data, q)
        links: dict[str, np.ndarray] = {}
        for link in self.link_names:
            if self.model.existFrame(link):
                links[link] = np.asarray(
                    self.data.oMf[self.model.getFrameId(link)].homogeneous,
                    dtype=np.float32,
                ).copy()
        links.setdefault(self.root_link, np.eye(4, dtype=np.float32))
        joints = {
            name: np.asarray(
                self.data.oMi[self._joint_ids[name]].translation,
                dtype=np.float32,
            ).copy()
            for name in self._joint_ids
        }
        return {"links": links, "joints": joints}

    def edges(self, pose: dict[str, object], names: tuple[str, ...]) -> list[tuple[np.ndarray, np.ndarray, str]]:
        links = pose["links"]
        result = []
        for name in names:
            joint = self.joints[name]
            result.append(
                (
                    np.asarray(links[joint.parent])[:3, 3],
                    np.asarray(links[joint.child])[:3, 3],
                    name,
                )
            )
        return result


class SyncDashboard:
    """Pygame HUD for the human-sync pose alignment step.

    Uses the same mesh views and corrective rotation arrows as BeingFlow.
    """

    BG = (8, 14, 18)
    PANEL = (15, 27, 32)
    PANEL_ALT = (11, 22, 27)
    BORDER = (45, 69, 75)
    GRID = (29, 49, 55)
    WHITE = (231, 239, 237)
    MUTED = (125, 153, 157)
    CYAN = (73, 211, 220)
    GLOVE = (75, 153, 167)
    GREEN = (75, 220, 137)
    AMBER = (244, 178, 72)
    RED = (244, 91, 91)

    def __init__(self, urdf_path: str | Path, width: int = 1200, height: int = 700):
        self.kinematics = UrdfKinematics(urdf_path)
        self.width, self.height = int(width), int(height)
        self._bounds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._pygame = None
        self._surface = None
        self._screen = None
        self._window = None
        self._fonts: dict[tuple[int, bool], object] = {}
        self._closed = False
        self._previous_focus = None
        self.model = mujoco.MjModel.from_xml_path(str(Path(__file__).with_name("assets") / "d1" / "mjcf" / "d1.xml"))
        self.current = mujoco.MjData(self.model)
        self.target = mujoco.MjData(self.model)
        self.hand_current = mujoco.MjData(self.model)
        self.hand_target = mujoco.MjData(self.model)
        self._renderers = {}
        self._hide_labels = False
        self._joint_ids_mesh = {name: self.model.joint(name).id for name in ARM_JOINTS + HAND_JOINTS}
        hand_root = self.model.body("flange").id
        for i in range(self.model.ngeom):
            body = int(self.model.geom_bodyid[i])
            while body and body != hand_root:
                body = int(self.model.body_parentid[body])
            self.model.geom_group[i] = 1 if body == hand_root else 0
            mesh = int(self.model.geom_dataid[i])
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_MESH, mesh) if mesh >= 0 else ''
            if name and (name.startswith('neck') or name == 'camera'):
                self.model.geom_group[i] = 2  # Head state is not supplied by this HUD.
            self.model.geom_rgba[i] = (.72, .76, .79, 1)
        self.model.vis.headlight.ambient[:] = (.4, .4, .4)
        self.model.vis.headlight.diffuse[:] = (.75, .75, .75)

    def _pg(self):
        if self._pygame is None:
            try:
                os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
                import pygame
            except ImportError as exc:
                raise RuntimeError(
                    "Human sync HUD needs pygame-ce; install it with "
                    "'.venv/bin/python -m pip install pygame-ce'"
                ) from exc
            pygame.font.init()
            self._pygame = pygame
        return self._pygame

    def _ensure_surface(self):
        pygame = self._pg()
        if self._surface is None:
            self._surface = pygame.Surface((self.width, self.height), depth=32)
        return self._surface

    def start(self) -> None:
        """Open the Pygame window; called lazily when human sync starts."""
        # SDL surface acceleration can bind a GLX context that conflicts with EGL.
        # Only the window blit is software; MuJoCo still renders meshes on the GPU.
        os.environ.setdefault("SDL_FRAMEBUFFER_ACCELERATION", "0")
        pygame = self._pg()
        self._previous_focus = _x11_focus_window()
        self._window = pygame.Window(
            "ALIGN POSE",
            size=(self.width, self.height),
            resizable=True,
            input_focus=True,
            mouse_focus=True,
        )
        self._screen = self._window.get_surface()
        self._closed = False
        self._ensure_surface().fill(self.BG)
        self._window.focus()
        self.present()

    def stop(self) -> None:
        for renderer in getattr(self, "_renderers", {}).values():
            renderer.close()
        if hasattr(self, "_renderers"):
            self._renderers.clear()
        window = getattr(self, "_window", None)
        if window is not None:
            window.destroy()
            self._window = None
        if self._pygame is not None and self._pygame.display.get_init():
            self._pygame.display.quit()
        self._screen = None
        _restore_x11_focus(self._previous_focus)
        self._previous_focus = None

    close = stop

    @staticmethod
    def _arm_xy(points: np.ndarray, view: str) -> np.ndarray:
        columns = {"Front": (0, 2), "Side": (1, 2), "Top": (0, 1)}[view]
        return points[:, columns]

    def _arm_points(self, pose: dict[str, object]) -> np.ndarray:
        links, joints = pose["links"], pose["joints"]
        return np.asarray(
            [
                np.asarray(links[self.kinematics.root_link])[:3, 3],
                *(joints[name] for name in ARM_JOINTS),
                np.asarray(links[self.kinematics.hand_root])[:3, 3],
            ],
            dtype=np.float32,
        )

    def _project_points(
        self,
        points: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
        key: str,
    ) -> np.ndarray:
        if key not in self._bounds:
            low, high = points.min(axis=0), points.max(axis=0)
            span = np.maximum(high - low, 0.05)
            margin = span * 0.20 + 0.02
            self._bounds[key] = (low - margin, high + margin)
        low, high = self._bounds[key]
        span = np.maximum(high - low, 1e-4)
        normalized = (points - low) / span
        return np.column_stack(
            (
                x + normalized[:, 0] * max(width - 1, 1),
                y + (1 - normalized[:, 1]) * max(height - 1, 1),
            )
        )

    @classmethod
    def _color(cls, error: float, threshold: float) -> tuple[int, int, int]:
        ratio = abs(float(error)) / max(float(threshold), 1e-6)
        if ratio <= 1:
            return cls.GREEN if ratio <= 0.5 else cls.AMBER
        return cls.RED

    @staticmethod
    def _points(points) -> list[tuple[int, int]]:
        return [tuple(np.round(point).astype(int)) for point in points]

    def _font(self, size: int, bold: bool = False):
        key = (int(size), bool(bold))
        if key not in self._fonts:
            self._fonts[key] = self._pg().font.SysFont(
                "DejaVu Sans", int(size), bold=bool(bold)
            )
        return self._fonts[key]

    def _text(
        self,
        surface,
        text: str,
        position: tuple[int, int],
        color,
        size: int = 16,
        bold: bool = False,
        align: str = "left",
    ) -> None:
        rendered = self._font(size, bold).render(str(text), True, color)
        rect = rendered.get_rect()
        if align == "center":
            rect.midtop = position
        elif align == "right":
            rect.right, rect.top = position
        else:
            rect.topleft = position
        surface.blit(rendered, rect)

    def _panel(
        self,
        surface,
        rect: tuple[int, int, int, int],
        fill,
        alpha: int,
        radius: int,
        outline,
    ) -> None:
        pygame = self._pg()
        x, y, width, height = rect
        overlay = pygame.Surface((width, height), pygame.SRCALPHA)
        pygame.draw.rect(
            overlay,
            (*fill, int(alpha)),
            overlay.get_rect(),
            border_radius=min(radius, width // 2, height // 2),
        )
        surface.blit(overlay, (x, y))
        if outline is not None:
            pygame.draw.rect(
                surface,
                outline,
                rect,
                width=1,
                border_radius=min(radius, width // 2, height // 2),
            )

    def _line(self, surface, points, color, width: int = 1, antialias: bool = True) -> None:
        pygame = self._pg()
        points = self._points(points)
        if len(points) < 2:
            return
        if antialias:
            pygame.draw.aalines(surface, color, False, points)
        if width > 1:
            pygame.draw.lines(surface, color, False, points, width)

    def _dashed(self, surface, points, color, width: int = 1, dash: int = 7) -> None:
        pygame = self._pg()
        for start, end in zip(points, points[1:]):
            start, end = np.asarray(start, dtype=np.float32), np.asarray(end, dtype=np.float32)
            vector = end - start
            length = float(np.linalg.norm(vector))
            if length < 1:
                continue
            unit = vector / length
            for offset in np.arange(0, length, dash * 2):
                a = start + unit * offset
                b = start + unit * min(offset + dash, length)
                pygame.draw.line(surface, color, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), width)

    def _rotation_arrow(
        self,
        surface,
        center: tuple[int, int],
        radius: int,
        positive: bool,
        color,
    ) -> None:
        pygame = self._pg()
        cx, cy = center
        # Draw the arc as points so its direction is unambiguous in Pygame's
        # screen coordinates: increasing angle is clockwise.
        sweep = np.pi * 1.25
        direction = 1.0 if positive else -1.0
        angles = -np.pi / 4 + direction * np.linspace(0.0, sweep, 28)
        points = np.column_stack(
            (cx + radius * np.cos(angles), cy + radius * np.sin(angles))
        )
        self._line(surface, points, color, 4)
        tip = points[-1]
        tangent = direction * np.asarray(
            (-np.sin(angles[-1]), np.cos(angles[-1])), dtype=np.float32
        )
        tangent /= max(float(np.linalg.norm(tangent)), 1e-6)
        back = tip - tangent * 13
        normal = np.asarray((-tangent[1], tangent[0])) * 8
        pygame.draw.polygon(
            surface,
            color,
            self._points((tip, back + normal, back - normal)),
        )

    def _linear_arrow(
        self,
        surface,
        center: tuple[int, int],
        positive: bool,
        kind: str,
        color,
        length: int,
    ) -> None:
        pygame = self._pg()
        sign = 1.0 if positive else -1.0
        if kind == "horizontal":
            vector = np.asarray((sign * length, 0.0), dtype=np.float32)
        else:
            vector = np.asarray((0.0, -sign * length), dtype=np.float32)
        origin = np.asarray(center, dtype=np.float32) - vector * 0.35
        tip = np.asarray(center, dtype=np.float32) + vector
        unit = vector / max(float(np.linalg.norm(vector)), 1e-6)
        back = tip - unit * 11
        normal = np.asarray((-unit[1], unit[0])) * 7
        pygame.draw.line(
            surface,
            color,
            tuple(np.round(origin).astype(int)),
            tuple(np.round(tip).astype(int)),
            4,
        )
        pygame.draw.polygon(surface, color, self._points((tip, back + normal, back - normal)))

    def _joint_direction_marker(
        self,
        surface,
        center: tuple[int, int],
        positive: bool,
        kind: str,
        color,
        radius: int,
    ) -> None:
        if kind == "rotation":
            self._rotation_arrow(surface, center, radius, positive, color)
        else:
            self._linear_arrow(surface, center, positive, kind, color, radius)

    def _direction_label(
        self,
        surface,
        center: tuple[int, int],
        text: str,
        color,
        bounds: tuple[int, int, int, int],
    ) -> None:
        x, y, width, height = bounds
        label_width = self._font(12, True).size(text)[0]
        if center[0] + label_width + 34 < x + width:
            position, align = (center[0] + 24, center[1] - 30), "left"
        else:
            position, align = (center[0] - 24, center[1] - 30), "right"
        self._panel(
            surface,
            (
                position[0] - 5 if align == "left" else position[0] - label_width - 5,
                position[1] - 3,
                label_width + 10,
                22,
            ),
            self.PANEL,
            235,
            5,
            color,
        )
        self._text(surface, text, position, color, 12, True, align)

    def _grid(self, surface, rect, horizontal: str, vertical: str) -> None:
        pygame = self._pg()
        x, y, width, height = rect
        for fraction in (0.33, 0.66):
            gx, gy = int(x + width * fraction), int(y + height * fraction)
            pygame.draw.line(surface, self.GRID, (gx, y), (gx, y + height), 1)
            pygame.draw.line(surface, self.GRID, (x, gy), (x + width, gy), 1)
        pygame.draw.line(surface, self.GRID, (x, y + height // 2), (x + width, y + height // 2), 1)
        self._text(surface, horizontal, (x + width - 8, y + height - 19), self.MUTED, 11, align="right")
        self._text(surface, vertical, (x + 8, y + 7), self.MUTED, 11)

    def _arm_view(self, surface, current, target, arm_error, threshold, view, rect, title_size):
        azimuth, elevation, axes = {
            'Front': (90, 0, 'X / Z'),
            'Side': (0, 0, 'Y / Z'),
            'Top': (0, -90, 'X / Y'),
        }[view]
        label = {'Front': 'SIDE', 'Side': 'BACK', 'Top': 'TOP'}[view]
        current_xy, target_xy, basis = self._mesh_view(surface, rect, f'{label}   {axes}' if view == 'Front' else '', azimuth, elevation, mirror=view == 'Front')
        self._joint_status(surface, rect, current_xy, target_xy, basis, arm_error, threshold, show_labels=view == "Front")

    @staticmethod
    def _finger_points(base: np.ndarray, length: float, splay: float, flex: float) -> np.ndarray:
        flex = float(np.clip(flex, 0, 1))
        a0 = -np.pi / 2 + splay
        a1, a2 = a0 + flex * 0.82, a0 + flex * 1.58
        result = [base]
        for angle, part in ((a0, 0.38), (a1, 0.32), (a2, 0.25)):
            result.append(
                result[-1]
                + length * part * np.asarray((np.cos(angle), np.sin(angle)))
            )
        return np.asarray(result, dtype=np.float32)

    def _hand_skeleton(
        self, state: np.ndarray, x: int, y: int, width: int, height: int
    ) -> dict[str, np.ndarray]:
        state = np.asarray(state, dtype=np.float32).reshape(6)
        # Use one scale for both axes.  Independent width/height factors make
        # the hand look unnaturally stretched in the tall right-hand panel.
        scale = min(width, height) * 0.5
        center = np.asarray((x + width * 0.39, y + height * 0.57), dtype=np.float32)
        tilt = 0.10
        rotation = np.asarray(
            ((np.cos(tilt), -np.sin(tilt)), (np.sin(tilt), np.cos(tilt))),
            dtype=np.float32,
        )

        def transform(points: np.ndarray) -> np.ndarray:
            return np.asarray(points) @ rotation.T + center

        bases = np.asarray(
            [
                (-scale * 0.19, -scale * 0.18),
                (-scale * 0.065, -scale * 0.20),
                (scale * 0.065, -scale * 0.19),
                (scale * 0.18, -scale * 0.15),
            ],
            dtype=np.float32,
        )
        result = {
            label: transform(
                self._finger_points(base, scale * length, splay, state[2 + index])
            )
            for index, (label, base, length, splay) in enumerate(
                zip(
                    ("Index", "Middle", "Ring", "Pinky"),
                    bases,
                    (0.57, 0.63, 0.59, 0.51),
                    (-0.06, -0.02, 0.02, 0.06),
                )
            )
        }
        thumb_base = np.asarray((-scale * 0.25, scale * 0.06), dtype=np.float32)
        thumb_flex = float(np.clip(0.55 * state[0] + 0.45 * state[1], 0, 1))
        result["Thumb"] = transform(
            self._finger_points(
                thumb_base,
                scale * 0.40,
                1.5 * np.pi - 0.35 * thumb_flex,
                thumb_flex,
            )
        )
        wrist = np.asarray((scale * 0.02, scale * 0.20), dtype=np.float32)
        result["Palm"] = transform(
            np.asarray(
                [(wrist, base) for base in tuple(bases) + (thumb_base,)],
                dtype=np.float32,
            )
        )
        result["Wrist"] = transform(
            np.asarray((wrist, (scale * 0.72, scale * 0.25)))
        )
        return result

    def _hand_skeleton_draw(self, surface, skeleton, color, dashed: bool) -> None:
        for segment in skeleton["Palm"]:
            if dashed:
                self._dashed(surface, segment, color, 1, 5)
            else:
                self._line(surface, segment, color, 3)
        for label in ("Index", "Middle", "Ring", "Pinky", "Thumb"):
            points = skeleton[label]
            if dashed:
                self._dashed(surface, points, color, 1, 5)
            else:
                self._line(surface, points, color, 3)
        if dashed:
            self._dashed(surface, skeleton["Wrist"], color, 1, 5)
        else:
            self._line(surface, skeleton["Wrist"], color, 3)

    def _hand_view(self, surface, current_state, target_state, hand_error, threshold, rect):
        # Isolate finger articulation from arm/wrist motion, like the legacy hand view.
        for data, state in ((self.hand_current, current_state), (self.hand_target, target_state)):
            self._pose(data, np.concatenate((np.zeros(6), np.asarray(state)[6:])))
        # Zero-pose dorsal normal is (-X + Z); fingers point up, thumb to the left.
        self._mesh_view(surface, rect, 'HAND   WHITE:ROBOT  CYAN:TARGET', 0, -45, hand=True)

    def _footer(
        self,
        surface,
        arm_error: np.ndarray,
        hand_error: np.ndarray,
        arm_threshold: float,
        hand_threshold: float,
        stable_frames: int,
        required_frames: int,
    ) -> None:
        pygame = self._pg()
        x, y, width, height = 18, self.height - 88, self.width - 36, 70
        self._panel(surface, (x, y, width, height), self.PANEL, 245, 10, self.BORDER)
        errors = np.concatenate((arm_error, hand_error))
        thresholds = np.asarray([arm_threshold] * 6 + [hand_threshold] * 6)
        good = np.abs(errors) <= thresholds
        in_range = int(np.sum(good))
        self._text(surface, "ALIGNMENT", (x + 18, y + 12), self.MUTED, 11, True)
        self._text(surface, "ARM", (x + 21, y + 31), self.MUTED, 10)
        self._text(surface, "HAND", (x + 195, y + 31), self.MUTED, 10)
        for index, is_good in enumerate(good):
            center = (x + 35 + index * 28, y + 52)
            color = self.GREEN if is_good else self._color(errors[index], thresholds[index])
            pygame.draw.circle(surface, color, center, 6)
            if not is_good:
                pygame.draw.circle(surface, self.PANEL, center, 2)
        status = "READY" if in_range == 12 else "SYNCING"
        status_color = self.GREEN if status == "READY" else self.AMBER
        self._text(surface, f"{in_range:02d} / 12  IN RANGE", (x + 370, y + 15), self.WHITE, 18, True)
        detail = f"STABLE  {min(stable_frames, required_frames):02d}/{required_frames:02d}"
        self._text(surface, detail, (x + 370, y + 42), self.MUTED, 12)
        hold_x = x + width - 235
        self._panel(surface, (hold_x, y + 13, 210, 43), status_color, 28, 9, status_color)
        pygame.draw.circle(surface, status_color, (hold_x + 18, y + 34), 5)
        self._text(surface, "HOLD", (hold_x + 31, y + 19), self.WHITE, 14, True)
        self._text(surface, status, (hold_x + 190, y + 19), status_color, 14, True, "right")

    def render(
        self,
        current_state: np.ndarray,
        target_state: np.ndarray,
        arm_error_deg: np.ndarray,
        hand_error: np.ndarray,
        arm_threshold_deg: float,
        hand_threshold: float,
        stable_frames: int = 0,
        required_frames: int = 3,
    ) -> np.ndarray:
        self._pose(self.current, current_state)
        self._pose(self.target, target_state)
        self._hide_labels = (stable_frames >= required_frames
                             and np.all(np.abs(arm_error_deg) <= arm_threshold_deg)
                             and np.all(np.abs(hand_error) <= hand_threshold))
        pygame = self._pg()
        surface = self._ensure_surface()
        current = self.kinematics.pose(current_state)
        target = self.kinematics.pose(target_state)
        arm_error_deg = np.asarray(arm_error_deg, dtype=np.float32).reshape(6)
        hand_error = np.asarray(hand_error, dtype=np.float32).reshape(6)
        surface.fill(self.BG)

        margin = 18
        gutter = 10
        content_y = margin
        content_bottom = self.height - 102
        content_height = max(content_bottom - content_y, 1)
        left_width = int(self.width * 0.72)
        arm_rect = (margin, content_y, left_width - margin * 2, content_height)
        hand_x = left_width - margin + gutter
        hand_rect = (
            hand_x,
            content_y,
            self.width - margin - hand_x,
            content_height,
        )
        inner_x, inner_y, inner_width, inner_height = arm_rect
        front_width = int(inner_width * 0.69)
        mini_width = inner_width - front_width - gutter
        mini_height = (inner_height - gutter) // 2
        self._arm_view(
            surface,
            current,
            target,
            arm_error_deg,
            arm_threshold_deg,
            "Front",
            (inner_x, inner_y, front_width, inner_height),
            17,
        )
        mini_x = inner_x + front_width + gutter
        self._arm_view(
            surface,
            current,
            target,
            arm_error_deg,
            arm_threshold_deg,
            "Side",
            (mini_x, inner_y, mini_width, mini_height),
            13,
        )
        self._arm_view(
            surface,
            current,
            target,
            arm_error_deg,
            arm_threshold_deg,
            "Top",
            (mini_x, inner_y + mini_height + gutter, mini_width, mini_height),
            13,
        )
        self._hand_view(
            surface,
            current_state,
            target_state,
            hand_error,
            hand_threshold,
            hand_rect,
        )
        self._footer(
            surface,
            arm_error_deg,
            hand_error,
            arm_threshold_deg,
            hand_threshold,
            stable_frames,
            required_frames,
        )

        if self._screen is not None and not self._closed:
            self.present()
        pixels = pygame.surfarray.array3d(surface)
        return np.ascontiguousarray(np.transpose(pixels, (1, 0, 2))[:, :, ::-1])

    def _pose(self, data, state):
        state = np.asarray(state, dtype=float)
        if state.shape != (12,) or not np.isfinite(state).all():
            raise ValueError("dashboard requires twelve finite joint values")
        data.qpos[:] = self.model.qpos0
        for name, value in zip(ARM_JOINTS, state[:6]):
            data.qpos[self.model.jnt_qposadr[self._joint_ids_mesh[name]]] = value
        for name, channel in zip(HAND_JOINTS, HAND_CHANNELS):
            joint = self._joint_ids_mesh[name]
            low, high = self.model.jnt_range[joint]
            data.qpos[self.model.jnt_qposadr[joint]] = low + np.clip(state[6 + channel], 0, 1) * (high - low)
        mujoco.mj_forward(self.model, data)

    def _mesh_view(self, surface, rect, title, azimuth, elevation, hand=False, mirror=False):
        x, y, width, height = rect
        self._panel(surface, rect, self.PANEL_ALT, 230, 10, self.BORDER)
        if title:
            self._text(surface, title, (x + 16, y + 13), self.CYAN, 13, True)
        w, h = width - 24, height - 48
        key = (w, h)
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self.model, width=w, height=h)
        renderer = self._renderers[key]
        option = mujoco.MjvOption()
        option.geomgroup[:] = 0
        option.geomgroup[1] = 1
        option.geomgroup[0] = not hand
        visible = self.model.geom_group == 1 if hand else self.model.geom_group < 2
        current, target = (self.hand_current, self.hand_target) if hand else (self.current, self.target)
        positions = np.vstack((current.geom_xpos[visible], target.geom_xpos[visible]))
        radii = np.tile(self.model.geom_rbound[visible], 2)[:, None]
        low, high = (positions - radii).min(axis=0), (positions + radii).max(axis=0)
        camera = mujoco.MjvCamera()
        self.model.vis.global_.orthographic = 1
        self.model.vis.global_.ipd = 0
        camera.lookat[:] = (low + high) / 2
        camera.azimuth, camera.elevation = azimuth, elevation
        az, el = np.deg2rad([azimuth, elevation])
        right = np.array([np.sin(az), -np.cos(az), 0])
        up = np.array([-np.cos(az) * np.sin(el), -np.sin(az) * np.sin(el), np.cos(el)])
        projected = positions @ np.stack((right, up), axis=1)
        extent = (projected + radii).max(axis=0) - (projected - radii).min(axis=0)
        self.model.vis.global_.fovy = max(extent[1], extent[0] * h / w, .1) * 1.12
        camera.distance = max(float(np.linalg.norm(high - low)) * 2, 1)
        renderer.update_scene(current, camera=camera, scene_option=option)
        scene = renderer.scene
        first = scene.ngeom
        mujoco.mjv_addGeoms(self.model, target, option, mujoco.MjvPerturb(),
                          mujoco.mjtCatBit.mjCAT_DYNAMIC, scene)
        for index in range(first, scene.ngeom):
            scene.geoms[index].rgba[:] = (.1, .85, .9, .28)
        scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        pixels = renderer.render()
        if mirror:
            pixels = pixels[:, ::-1].copy()
            right = -right
        pixels[np.all(pixels == 0, axis=2)] = self.BG
        pygame = self._pg()
        image = pygame.surfarray.make_surface(np.transpose(pixels, (1, 0, 2)))
        surface.blit(image, (x + 12, y + 36))
        points = np.vstack([
            data.xanchor[[self._joint_ids_mesh[name] for name in ARM_JOINTS]]
            for data in (self.current, self.target)
        ])
        offset = (points - camera.lookat) @ np.stack((right, up), axis=1)
        offset *= h / self.model.vis.global_.fovy
        screen = np.column_stack((x + 12 + w / 2 + offset[:, 0],
                                  y + 36 + h / 2 - offset[:, 1]))
        basis = np.stack((right, -up), axis=1) * h / self.model.vis.global_.fovy
        return screen[:6], screen[6:], basis

    def _joint_status(self, surface, rect, current_xy, target_xy, basis, errors, threshold, show_labels=True):
        pygame = self._pg()
        x, y, width, height = rect
        bounds = pygame.Rect(x + 14, y + 40, width - 28, height - 74)
        occupied = []
        labels = []
        for index, error in enumerate(errors):
            center = tuple(np.round(current_xy[index]).astype(int))
            target = tuple(np.round(target_xy[index]).astype(int))
            color = self.GREEN if abs(error) <= threshold else self.RED
            if abs(error) > threshold:
                self._joint_rotation_arrow(surface, center, index, -float(error), basis, color)
            pygame.draw.circle(surface, color, center, 6, 2)
        if self._hide_labels or not show_labels:
            return
        for index in np.argsort(current_xy[:, 1]):
            center = tuple(np.round(current_xy[index]).astype(int))
            target = tuple(np.round(target_xy[index]).astype(int))
            error = float(errors[index])
            good = abs(error) <= threshold
            color = self.GREEN if good else self.RED
            text = 'OK' if good else f'{abs(error):.1f}°'
            label_width = self._font(12, True).size(text)[0] + 10
            # Try nearby positions before moving a label, keeping overlapping joints legible.
            label = None
            for dy in (0, -22, 22, -44, 44, -66, 66, -88, 88):
                for left in (center[0] + 10, center[0] - label_width - 10):
                    candidate = pygame.Rect(left, center[1] - 10 + dy, label_width, 20).clamp(bounds)
                    if not any(candidate.inflate(4, 4).colliderect(other) for other in occupied):
                        label = candidate
                        break
                if label is not None:
                    break
            if label is None:
                label = candidate
            occupied.append(label)
            end = (label.left if label.centerx > center[0] else label.right, label.centery)
            pygame.draw.line(surface, color, center, end, 1)
            labels.append((label, text, color))
        # Paint tags last so another joint's marker or leader cannot obscure text.
        for label, text, color in labels:
            self._panel(surface, tuple(label), self.PANEL, 255, 4, color)
            self._text(surface, text, (label.x + 5, label.y + 2), color, 12, True)

    def _joint_rotation_arrow(self, surface, center, index, correction, basis, color):
        # Project a rotation around the real joint axis, not an arbitrary screen circle.
        axis = self.current.xaxis[self._joint_ids_mesh[ARM_JOINTS[index]]]
        reference = np.array([1., 0., 0.]) if abs(axis[0]) < .9 else np.array([0., 1., 0.])
        first = np.cross(axis, reference)
        first /= np.linalg.norm(first)
        second = np.cross(axis, first)
        angles = np.pi / 4 + np.sign(correction) * np.linspace(0, np.pi * 1.25, 40)
        radius = 19 / np.linalg.norm(basis[:, 0])
        ring = radius * (np.cos(angles)[:, None] * first + np.sin(angles)[:, None] * second)
        points = np.asarray(center) + ring @ basis
        self._line(surface, points, color, 2)
        tip = points[-1]
        tangent = tip - points[-3]
        length = np.linalg.norm(tangent)
        if length > 1e-6:
            tangent /= length
            normal = np.array([-tangent[1], tangent[0]]) * 4
            back = tip - tangent * 8
            self._pg().draw.polygon(surface, color, self._points((tip, back + normal, back - normal)))

    def present(self) -> None:
        if self._screen is None or self._closed:
            return
        pygame = self._pg()
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                self._closed = True
                self.stop()
                return
        if self._screen.get_size() == (self.width, self.height):
            self._screen.blit(self._surface, (0, 0))
        else:
            scaled = pygame.transform.smoothscale(self._surface, self._screen.get_size())
            self._screen.blit(scaled, (0, 0))
        if self._window is not None:
            self._window.flip()
        else:
            pygame.display.flip()


if __name__ == "__main__":
    raise SystemExit("Use SyncDashboard from the online runner.")
