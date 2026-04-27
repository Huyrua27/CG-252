from __future__ import annotations

import io
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from libs import transform as T

from .asset_catalog import (
    CLASS_NAMES,
    CLASS_TO_ID,
    AssetEntry,
    collect_reference_images,
    export_catalog_json,
    resolve_catalog,
)
from .mesh_loader import (
    MeshData,
    build_billboard_mesh,
    build_ground_plane_mesh,
    keep_upward_facing_triangles,
    load_obj,
    remove_planar_artifact_components,
    remove_tiny_high_material_components,
)
from .renderer import CameraParams, GLDatasetRenderer, LightParams, SceneInstance


@dataclass
class RuntimeAsset:
    entry: AssetEntry
    mesh_key: str
    center_x: float
    min_y: float
    center_z: float
    scale_vec: np.ndarray
    size_world: np.ndarray
    length_world: float
    base_rotation_deg: np.ndarray
    base_offset: np.ndarray
    surface_y: float


@dataclass
class FrameObject:
    instance_id: int
    class_id: int
    class_name: str
    asset_key: str
    position: np.ndarray
    yaw_deg: float
    extra_scale: float
    bbox_xywh: Optional[List[int]] = None
    visible_pixels: int = 0


@dataclass
class SequenceActor:
    instance_id: int
    asset_key: str
    class_name: str
    position: np.ndarray
    base_y: float
    yaw_deg: float
    extra_scale: float
    speed: float
    lane_x: float
    lateral_amp: float = 0.0
    lateral_freq: float = 0.0
    lateral_phase: float = 0.0
    bob_amp: float = 0.0
    bob_freq: float = 0.0
    bob_phase: float = 0.0
    wheel_radius: float = 0.0
    wheel_spin: float = 0.0
    wheel_slide: float = 0.0
    use_alpha_key: bool = False
    dynamic: bool = False
    respawn_min_z: float = -40.0
    respawn_max_z: float = -6.0
    direction_z: float = 1.0


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _to_np3(values: List[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def _rotation_xyz_deg(rot: np.ndarray) -> np.ndarray:
    rx = T.rotate(axis=(1.0, 0.0, 0.0), angle=float(rot[0]))
    ry = T.rotate(axis=(0.0, 1.0, 0.0), angle=float(rot[1]))
    rz = T.rotate(axis=(0.0, 0.0, 1.0), angle=float(rot[2]))
    return (rx @ ry @ rz).astype(np.float32)


def _normalize3(vec: np.ndarray, fallback: Tuple[float, float, float] = (0.0, 0.0, -1.0)) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(arr))
    if n < 1e-8:
        return np.asarray(fallback, dtype=np.float32)
    return (arr / n).astype(np.float32)


def _rotate_vec_axis(vec: np.ndarray, axis: np.ndarray, angle_deg: float) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32)
    k = _normalize3(axis)
    theta = math.radians(float(angle_deg))
    c = float(math.cos(theta))
    s = float(math.sin(theta))
    cross = np.cross(k, v)
    dot = float(np.dot(k, v))
    return (v * c + cross * s + k * dot * (1.0 - c)).astype(np.float32)


def _sun_angles_from_direction(sun_direction: np.ndarray) -> Tuple[float, float]:
    # `sun_direction` points from sun -> surface; convert to scene -> sun.
    scene_to_sun = _normalize3(-np.asarray(sun_direction, dtype=np.float32))
    azimuth = math.degrees(math.atan2(float(scene_to_sun[0]), float(-scene_to_sun[2])))
    elevation = math.degrees(math.asin(float(np.clip(scene_to_sun[1], -1.0, 1.0))))
    return float(azimuth), float(elevation)


def _sun_direction_from_angles(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    az = math.radians(float(azimuth_deg))
    el = math.radians(float(elevation_deg))
    scene_to_sun = np.asarray(
        [
            math.sin(az) * math.cos(el),
            math.sin(el),
            -math.cos(az) * math.cos(el),
        ],
        dtype=np.float32,
    )
    return _normalize3(-scene_to_sun)


def _compose_model_matrix(
    position: np.ndarray,
    yaw_deg: float,
    local_center: np.ndarray,
    base_rotation_deg: np.ndarray,
    base_scale: np.ndarray,
    base_offset: np.ndarray,
    extra_scale: float = 1.0,
) -> np.ndarray:
    center_fix = T.translate(-local_center[0], -local_center[1], -local_center[2]).astype(np.float32)
    yaw_rot = T.rotate(axis=(0.0, 1.0, 0.0), angle=float(yaw_deg)).astype(np.float32)
    base_rot = _rotation_xyz_deg(base_rotation_deg)
    scale = T.scale(base_scale * float(extra_scale)).astype(np.float32)
    translation = T.translate(position + base_offset).astype(np.float32)
    return (translation @ yaw_rot @ base_rot @ scale @ center_fix).astype(np.float32)


def _randint_pair(rng: np.random.Generator, pair: List[int]) -> int:
    low, high = int(pair[0]), int(pair[1])
    return int(rng.integers(low, high + 1))


def _write_image_unicode(path: Path, image: np.ndarray) -> None:
    suffix = path.suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg"}:
        raise ValueError(f"Unsupported image format for path: {path}")
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise RuntimeError(f"cv2.imencode failed for: {path}")
    path.write_bytes(encoded.tobytes())


def _write_npy_unicode(path: Path, array: np.ndarray) -> None:
    """
    Robust npy writer on Windows/unicode paths.
    Avoid NumPy fast-path `tofile` that can fail with partial-write OSError.
    """
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array), allow_pickle=False)
    path.write_bytes(buffer.getvalue())


def _instance_map_to_color(instance_map: np.ndarray) -> np.ndarray:
    """
    Convert integer instance IDs to a vivid RGB visualization image.
    Background ID 0 stays black.
    """
    ids = instance_map.astype(np.uint32)
    vis = np.zeros((ids.shape[0], ids.shape[1], 3), dtype=np.uint8)
    fg = ids > 0
    if not np.any(fg):
        return vis

    # Deterministic pseudo-palette from instance id.
    r = ((ids * 53) % 253).astype(np.uint8)
    g = ((ids * 97) % 251).astype(np.uint8)
    b = ((ids * 193) % 249).astype(np.uint8)

    # Lift dark colors so masks are readable in normal image viewers.
    r = np.where(fg, np.maximum(r, 28), 0).astype(np.uint8)
    g = np.where(fg, np.maximum(g, 28), 0).astype(np.uint8)
    b = np.where(fg, np.maximum(b, 28), 0).astype(np.uint8)

    vis[:, :, 0] = r
    vis[:, :, 1] = g
    vis[:, :, 2] = b
    return vis


def _class_color_bgr(class_id: int) -> Tuple[int, int, int]:
    cid = int(class_id)
    b = int((cid * 89 + 61) % 255)
    g = int((cid * 137 + 97) % 255)
    r = int((cid * 53 + 173) % 255)
    return (max(b, 48), max(g, 48), max(r, 48))


def _create_video_writer(path: Path, width: int, height: int, fps: int) -> Tuple[cv2.VideoWriter, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(width), int(height)),
    )
    if writer.isOpened():
        return writer, path

    fallback = path.with_suffix(".avi")
    writer = cv2.VideoWriter(
        str(fallback),
        cv2.VideoWriter_fourcc(*"MJPG"),
        float(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open video writer for: {path}")
    return writer, fallback


class BTL2DatasetGenerator:
    def __init__(
        self,
        sample_root: Path,
        output_root: Path,
        width: int,
        height: int,
        scene_config_path: Path,
        max_faces: Optional[int] = 0,
        rgb_source: str = "mesh",
        gaussian_rgb_dir: Optional[Path] = None,
        gaussian_blend: float = 1.0,
        fps: int = 24,
        scene_mode: str = "sequence",
        write_video: bool = True,
        time_of_day: str = "day",
        camera_view: str = "forward",
        sun_azimuth_deg: Optional[float] = None,
        sun_elevation_deg: Optional[float] = None,
        sun_intensity: Optional[float] = None,
        fill_intensity: Optional[float] = None,
        shadow_strength: Optional[float] = None,
        disable_shadows: bool = False,
        street_light_intensity: Optional[float] = None,
        street_light_count: int = 8,
        traffic_flow: Optional[str] = None,
    ) -> None:
        self.sample_root = sample_root.resolve()
        self.output_root = output_root.resolve()
        self.width = int(width)
        self.height = int(height)
        self.rgb_source = str(rgb_source)
        self.gaussian_rgb_dir = gaussian_rgb_dir.resolve() if gaussian_rgb_dir is not None else None
        self.gaussian_blend = float(np.clip(gaussian_blend, 0.0, 1.0))
        self.fps = max(1, int(fps))
        self.scene_mode = str(scene_mode)
        self.write_video = bool(write_video)
        self.time_of_day = str(time_of_day).lower()
        self.camera_view = str(camera_view).lower()
        self.sun_azimuth_deg = (None if sun_azimuth_deg is None else float(sun_azimuth_deg))
        self.sun_elevation_deg = (None if sun_elevation_deg is None else float(sun_elevation_deg))
        self.sun_intensity_override = (None if sun_intensity is None else float(sun_intensity))
        self.fill_intensity_override = (None if fill_intensity is None else float(fill_intensity))
        self.shadow_strength_override = (None if shadow_strength is None else float(shadow_strength))
        self.disable_shadows = bool(disable_shadows)
        self.street_light_intensity_override = (None if street_light_intensity is None else float(street_light_intensity))
        self.street_light_count = max(0, int(street_light_count))
        self._gaussian_rgb_hits = 0
        self._gaussian_rgb_misses = 0
        self.traffic_flow = (None if traffic_flow is None else str(traffic_flow).lower())

        self._time_of_day_presets: Dict[str, Dict[str, object]] = {
            "day": {
                "sun_color": [1.0, 0.98, 0.95],
                "fill_color": [0.62, 0.68, 0.75],
                "sky_color": [0.60, 0.73, 0.90],
                "ambient_strength": 0.30,
                "sun_intensity": 1.20,
                "fill_intensity": 0.95,
                "shadow_strength": 0.70,
                "street_light_intensity": 0.0,
                "street_light_color": [1.00, 0.90, 0.68],
            },
            "sunset": {
                "sun_color": [1.0, 0.76, 0.57],
                "fill_color": [0.58, 0.50, 0.62],
                "sky_color": [0.93, 0.63, 0.43],
                "ambient_strength": 0.24,
                "sun_intensity": 1.05,
                "fill_intensity": 0.92,
                "shadow_strength": 0.64,
                "street_light_intensity": 0.35,
                "street_light_color": [1.00, 0.88, 0.64],
            },
            "night": {
                "sun_color": [0.14, 0.18, 0.28],
                "fill_color": [0.44, 0.52, 0.70],
                "sky_color": [0.06, 0.08, 0.15],
                "ambient_strength": 0.16,
                "sun_intensity": 0.18,
                "fill_intensity": 1.34,
                "shadow_strength": 0.28,
                "street_light_intensity": 2.35,
                "street_light_color": [1.00, 0.86, 0.60],
            },
        }
        self._camera_profiles: Dict[str, Dict[str, object]] = {
            "forward": {"yaw": 0.0, "pitch": 0.0, "eye_offset": [0.0, 0.0, 0.0]},
            "left": {"yaw": -32.0, "pitch": 0.0, "eye_offset": [0.0, 0.0, 0.0]},
            "right": {"yaw": 32.0, "pitch": 0.0, "eye_offset": [0.0, 0.0, 0.0]},
            "rear": {"yaw": 180.0, "pitch": 0.0, "eye_offset": [0.0, 0.0, 0.0]},
            "bird": {"yaw": 0.0, "pitch": -30.0, "eye_offset": [0.0, 5.8, 0.0]},
        }

        camera_aliases = {
            "front": "forward",
            "back": "rear",
            "behind": "rear",
            "truoc": "forward",
            "sau": "rear",
            "trai": "left",
            "phai": "right",
        }
        self.camera_view = camera_aliases.get(self.camera_view, self.camera_view)

        if self.rgb_source not in {"mesh", "gaussian_hybrid"}:
            raise ValueError(f"Unsupported rgb_source: {self.rgb_source}")
        if self.scene_mode not in {"sequence", "random"}:
            raise ValueError(f"Unsupported scene_mode: {self.scene_mode}")
        if self.rgb_source == "gaussian_hybrid" and self.gaussian_rgb_dir is None:
            raise ValueError("gaussian_rgb_dir is required when rgb_source='gaussian_hybrid'")
        if self.time_of_day not in self._time_of_day_presets:
            raise ValueError(f"Unsupported time_of_day: {self.time_of_day}")
        if self.camera_view not in self._camera_profiles:
            raise ValueError(f"Unsupported camera_view: {self.camera_view}")

        self.scene_cfg = _load_json(scene_config_path.resolve())
        scene_flow = str(self.scene_cfg.get("scene", {}).get("traffic_direction", "into_scene")).lower()
        if self.traffic_flow is None:
            self.traffic_flow = scene_flow
        if self.traffic_flow not in {"into_scene", "toward_camera"}:
            raise ValueError(f"Unsupported traffic_flow: {self.traffic_flow}")
        self.traffic_direction_sign = (-1.0 if self.traffic_flow == "into_scene" else 1.0)

        self.catalog = resolve_catalog(self.sample_root, override_max_faces=max_faces)
        self.assets: Dict[str, RuntimeAsset] = {}

        shaders_dir = self.sample_root / "btl2_sys" / "shaders"
        self.renderer = GLDatasetRenderer(self.width, self.height, shaders_dir=shaders_dir)
        self._load_assets()
        self.road_surface_y = (
            float(np.clip(self.assets["road_street"].surface_y, 0.0, 0.35))
            if "road_street" in self.assets
            else 0.0
        )
        self._background_instances = self._build_background_instances()

        base_camera = CameraParams(
            eye=_to_np3(self.scene_cfg["camera"]["eye"]),
            target=_to_np3(self.scene_cfg["camera"]["target"]),
            up=_to_np3(self.scene_cfg["camera"]["up"]),
            fovy=float(self.scene_cfg["camera"]["fovy"]),
            near=float(self.scene_cfg["camera"]["near"]),
            far=float(self.scene_cfg["camera"]["far"]),
        )
        self.camera = self._configure_camera(base_camera)

        lighting_cfg = self.scene_cfg["lighting"]
        street_positions = self._build_street_light_positions(
            count=self.street_light_count,
            height=float(lighting_cfg.get("street_light_height", 6.4)),
        )
        base_light = LightParams(
            sun_direction=_to_np3(self.scene_cfg["lighting"]["sun_direction"]),
            sun_color=_to_np3(self.scene_cfg["lighting"]["sun_color"]),
            fill_position=_to_np3(self.scene_cfg["lighting"]["fill_position"]),
            fill_color=_to_np3(self.scene_cfg["lighting"]["fill_color"]),
            sky_color=_to_np3(lighting_cfg.get("sky_color", [0.60, 0.73, 0.90])),
            ambient_strength=float(lighting_cfg.get("ambient_strength", 0.30)),
            sun_intensity=float(lighting_cfg.get("sun_intensity", 1.20)),
            fill_intensity=float(lighting_cfg.get("fill_intensity", 0.95)),
            shadow_strength=float(lighting_cfg.get("shadow_strength", 0.70)),
            shadow_enabled=(not bool(lighting_cfg.get("disable_shadows", False))),
            street_light_positions=street_positions,
            street_light_color=_to_np3(lighting_cfg.get("street_light_color", [1.0, 0.90, 0.68])),
            street_light_intensity=float(lighting_cfg.get("street_light_intensity", 0.0)),
        )
        self.light = self._configure_lighting(base_light)

    def _build_street_light_positions(self, count: int, height: float = 6.4) -> np.ndarray:
        n = max(0, int(count))
        if n <= 0:
            return np.zeros((0, 3), dtype=np.float32)

        side_x = [9.8, -9.8]
        z_values = np.linspace(18.0, -86.0, num=max(n, 2), dtype=np.float32)
        points: List[np.ndarray] = []
        for i, z in enumerate(z_values.tolist()):
            x = side_x[i % 2]
            z_shift = -2.0 if (i % 2 == 0) else 2.0
            points.append(np.asarray([x, float(height), float(z + z_shift)], dtype=np.float32))
            if len(points) >= n:
                break
        return np.vstack(points).astype(np.float32) if points else np.zeros((0, 3), dtype=np.float32)

    def _configure_camera(self, base: CameraParams) -> CameraParams:
        profile = self._camera_profiles[self.camera_view]
        yaw = float(profile.get("yaw", 0.0))
        pitch = float(profile.get("pitch", 0.0))
        eye_offset = np.asarray(profile.get("eye_offset", [0.0, 0.0, 0.0]), dtype=np.float32)

        base_eye = np.asarray(base.eye, dtype=np.float32)
        base_target = np.asarray(base.target, dtype=np.float32)
        up = _normalize3(np.asarray(base.up, dtype=np.float32), fallback=(0.0, 1.0, 0.0))
        dist = float(np.linalg.norm(base_target - base_eye))
        dist = max(dist, 1e-3)

        forward = _normalize3(base_target - base_eye)
        if abs(yaw) > 1e-6:
            forward = _rotate_vec_axis(forward, up, yaw)
        right = _normalize3(np.cross(forward, up), fallback=(1.0, 0.0, 0.0))
        if abs(pitch) > 1e-6:
            forward = _rotate_vec_axis(forward, right, pitch)

        eye = (base_eye + eye_offset).astype(np.float32)
        target = (eye + forward * dist).astype(np.float32)
        return CameraParams(
            eye=eye,
            target=target,
            up=up,
            fovy=float(base.fovy),
            near=float(base.near),
            far=float(base.far),
        )

    def _configure_lighting(self, base: LightParams) -> LightParams:
        preset = self._time_of_day_presets[self.time_of_day]
        if self.time_of_day == "day":
            sun_color = np.asarray(base.sun_color, dtype=np.float32)
            fill_color = np.asarray(base.fill_color, dtype=np.float32)
            sky_color = np.asarray(base.sky_color, dtype=np.float32)
            ambient_strength = float(base.ambient_strength)
            sun_intensity = float(base.sun_intensity)
            fill_intensity = float(base.fill_intensity)
            shadow_strength = float(base.shadow_strength)
            street_light_intensity = float(base.street_light_intensity)
            street_light_color = np.asarray(base.street_light_color, dtype=np.float32)
        else:
            sun_color = _to_np3(preset["sun_color"])
            fill_color = _to_np3(preset["fill_color"])
            sky_color = _to_np3(preset["sky_color"])
            ambient_strength = float(preset["ambient_strength"])
            sun_intensity = float(preset["sun_intensity"])
            fill_intensity = float(preset["fill_intensity"])
            shadow_strength = float(preset["shadow_strength"])
            street_light_intensity = float(preset["street_light_intensity"])
            street_light_color = _to_np3(preset["street_light_color"])

        shadow_enabled = bool(base.shadow_enabled) and (not self.disable_shadows)
        if self.sun_intensity_override is not None:
            sun_intensity = float(self.sun_intensity_override)
        if self.fill_intensity_override is not None:
            fill_intensity = float(self.fill_intensity_override)
        if self.shadow_strength_override is not None:
            shadow_strength = float(self.shadow_strength_override)
        if self.street_light_intensity_override is not None:
            street_light_intensity = float(self.street_light_intensity_override)

        street_positions = np.asarray(base.street_light_positions, dtype=np.float32).reshape(-1, 3)
        if self.street_light_count > 0 and street_positions.shape[0] > self.street_light_count:
            street_positions = street_positions[: self.street_light_count]
        elif self.street_light_count <= 0:
            street_positions = np.zeros((0, 3), dtype=np.float32)

        sun_azimuth, sun_elevation = _sun_angles_from_direction(base.sun_direction)
        if self.time_of_day == "sunset":
            sun_elevation = min(sun_elevation, 12.0)
        elif self.time_of_day == "night":
            sun_elevation = -8.0
        if self.sun_azimuth_deg is not None:
            sun_azimuth = float(self.sun_azimuth_deg)
        if self.sun_elevation_deg is not None:
            sun_elevation = float(self.sun_elevation_deg)
        sun_direction = _sun_direction_from_angles(sun_azimuth, sun_elevation)

        fill_position = np.asarray(base.fill_position, dtype=np.float32).copy()
        if self.time_of_day == "night":
            fill_position = (np.asarray(self.camera.eye, dtype=np.float32) + np.asarray([0.0, 3.2, 2.6], dtype=np.float32))

        return LightParams(
            sun_direction=sun_direction,
            sun_color=sun_color.astype(np.float32),
            fill_position=fill_position.astype(np.float32),
            fill_color=fill_color.astype(np.float32),
            sky_color=sky_color.astype(np.float32),
            ambient_strength=float(np.clip(ambient_strength, 0.04, 1.2)),
            sun_intensity=float(np.clip(sun_intensity, 0.0, 4.0)),
            fill_intensity=float(np.clip(fill_intensity, 0.0, 4.0)),
            shadow_strength=float(np.clip(shadow_strength, 0.0, 1.0)),
            shadow_enabled=shadow_enabled,
            street_light_positions=street_positions.astype(np.float32),
            street_light_color=street_light_color.astype(np.float32),
            street_light_intensity=float(np.clip(street_light_intensity, 0.0, 8.0)),
        )

    def _vehicle_lane_groups(self) -> Tuple[List[float], List[float], float, float]:
        lanes_cfg = [float(v) for v in self.scene_cfg["scene"]["lanes_x"]]
        lanes_unique = sorted({float(v) for v in lanes_cfg})
        neg_lanes = [x for x in lanes_unique if x < -0.35]
        pos_lanes = [x for x in lanes_unique if x > 0.35]

        if not neg_lanes or not pos_lanes:
            mid = float(np.median(np.asarray(lanes_unique, dtype=np.float32)))
            neg_lanes = [x for x in lanes_unique if x <= mid]
            pos_lanes = [x for x in lanes_unique if x > mid]
            if not pos_lanes:
                pos_lanes = neg_lanes[:]
            if not neg_lanes:
                neg_lanes = pos_lanes[:]

        neg_dir = float(self.traffic_direction_sign)
        pos_dir = float(-self.traffic_direction_sign)
        return neg_lanes, pos_lanes, neg_dir, pos_dir

    @staticmethod
    def _yaw_from_direction(direction_z: float, jitter_deg: float = 0.0, rng: Optional[np.random.Generator] = None) -> float:
        base = 180.0 if float(direction_z) < 0.0 else 0.0
        if rng is None or jitter_deg <= 0.0:
            return float(base)
        return float(base + float(rng.uniform(-jitter_deg, jitter_deg)))

    def _vehicle_actor_length(self, actor: SequenceActor) -> float:
        runtime = self.assets.get(actor.asset_key)
        if runtime is None:
            return 4.0
        return float(max(0.8, runtime.length_world * float(actor.extra_scale)))

    def _enforce_vehicle_lane_spacing(
        self,
        actors: List[SequenceActor],
        gap_buffer: float = 1.4,
    ) -> None:
        lane_groups: Dict[Tuple[int, int], List[SequenceActor]] = {}
        for actor in actors:
            if not actor.dynamic or actor.class_name not in {"car", "truck"}:
                continue
            dir_key = 1 if float(actor.direction_z) > 0.0 else -1
            lane_key = int(round(float(actor.lane_x) * 1000.0))
            lane_groups.setdefault((dir_key, lane_key), []).append(actor)

        for lane_actors in lane_groups.values():
            lane_actors.sort(key=lambda a: float(a.direction_z) * float(a.position[2]))
            for idx in range(1, len(lane_actors)):
                prev = lane_actors[idx - 1]
                curr = lane_actors[idx]
                prev_progress = float(prev.direction_z) * float(prev.position[2])
                curr_progress = float(curr.direction_z) * float(curr.position[2])
                min_gap = 0.5 * (self._vehicle_actor_length(prev) + self._vehicle_actor_length(curr)) + float(gap_buffer)
                if curr_progress - prev_progress >= min_gap:
                    continue
                target_progress = prev_progress + min_gap
                curr.position[2] = np.float32(target_progress / float(curr.direction_z))

    def _read_image_unicode_rgb(self, path: Path) -> Optional[np.ndarray]:
        if not path.exists():
            return None
        raw = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.ndim == 2:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return rgb

    def _choose_rgb_output(self, stem: str, mesh_rgb: np.ndarray) -> np.ndarray:
        if self.rgb_source != "gaussian_hybrid" or self.gaussian_rgb_dir is None:
            return mesh_rgb

        png_path = self.gaussian_rgb_dir / f"{stem}.png"
        jpg_path = self.gaussian_rgb_dir / f"{stem}.jpg"
        gauss = self._read_image_unicode_rgb(png_path)
        if gauss is None:
            gauss = self._read_image_unicode_rgb(jpg_path)
        if gauss is None:
            self._gaussian_rgb_misses += 1
            return mesh_rgb

        if gauss.shape[0] != self.height or gauss.shape[1] != self.width:
            gauss = cv2.resize(gauss, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        self._gaussian_rgb_hits += 1
        if self.gaussian_blend >= 0.999:
            return gauss.astype(np.uint8)
        if self.gaussian_blend <= 0.001:
            return mesh_rgb

        blended = (
            mesh_rgb.astype(np.float32) * (1.0 - self.gaussian_blend)
            + gauss.astype(np.float32) * self.gaussian_blend
        )
        return np.clip(blended, 0.0, 255.0).astype(np.uint8)

    def close(self) -> None:
        self.renderer.close()

    def _load_assets(self) -> None:
        for key, entry in self.catalog.items():
            mesh: MeshData
            if entry.obj_relpath is not None:
                mesh = load_obj(
                    self.sample_root / entry.obj_relpath,
                    max_faces=entry.max_faces,
                    keep_largest_component=entry.keep_largest_component,
                    remove_outliers=entry.remove_outliers,
                )
                if key == "car_audi_r8":
                    mesh = remove_tiny_high_material_components(
                        mesh,
                        material_name_tokens=("lightsource",),
                        max_component_faces=16,
                        min_height_quantile=0.80,
                    )
                    mesh = remove_planar_artifact_components(
                        mesh,
                        max_component_faces=8,
                        max_thickness_ratio=0.06,
                        min_major_extent=0.16,
                    )
                elif key == "building_city":
                    mesh = remove_planar_artifact_components(
                        mesh,
                        max_component_faces=1200,
                        max_thickness_ratio=0.10,
                        min_major_extent=0.20,
                    )
                elif key == "road_street":
                    mesh = keep_upward_facing_triangles(mesh, min_up_dot=0.85)
            elif entry.billboard_texture_relpath is not None:
                mesh = build_billboard_mesh(self.sample_root / entry.billboard_texture_relpath, name=key)
            else:
                raise ValueError(f"Asset '{key}' has no mesh source")

            # Force specific textures when source MTL is incomplete.
            if entry.forced_texture_relpaths:
                forced_paths = [
                    (self.sample_root / rel).resolve()
                    for rel in entry.forced_texture_relpaths
                    if (self.sample_root / rel).exists()
                ]
                if forced_paths:
                    primary_tex = forced_paths[0]
                    for mat in mesh.materials:
                        if mat.texture_path is None:
                            mat.texture_path = primary_tex
                            # Forced albedo should drive base color directly.
                            mat.kd = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
                            mat.ka = np.maximum(mat.ka, np.asarray([0.08, 0.08, 0.08], dtype=np.float32))

            ext = np.maximum(mesh.extents, 1e-4)
            if entry.scale_mode == "fixed" and entry.fixed_scale is not None:
                scale_vec = np.asarray(entry.fixed_scale, dtype=np.float32)
            elif entry.scale_mode == "height":
                s = float(entry.target_size) / float(ext[1])
                scale_vec = np.asarray([s, s, s], dtype=np.float32)
            else:
                length = max(float(ext[0]), float(ext[2]))
                s = float(entry.target_size) / max(length, 1e-5)
                scale_vec = np.asarray([s, s, s], dtype=np.float32)

            center_x = float((mesh.bbox_min[0] + mesh.bbox_max[0]) * 0.5)
            min_y = float(mesh.bbox_min[1])
            center_z = float((mesh.bbox_min[2] + mesh.bbox_max[2]) * 0.5)
            size_world = (ext * scale_vec).astype(np.float32)
            length_world = float(max(size_world[0], size_world[2]))

            # Roads are volumetric meshes, so median-Y sits inside thickness.
            # Use an upper percentile as the drivable surface reference.
            y_ref_pct = 95.0 if entry.category == "road" else 50.0
            surface_y = float((np.percentile(mesh.vertices[:, 1], y_ref_pct) - min_y) * float(scale_vec[1]))

            self.renderer.register_mesh(key, mesh)
            self.assets[key] = RuntimeAsset(
                entry=entry,
                mesh_key=key,
                center_x=center_x,
                min_y=min_y,
                center_z=center_z,
                scale_vec=scale_vec,
                size_world=size_world,
                length_world=length_world,
                base_rotation_deg=np.asarray(entry.base_rotation_deg, dtype=np.float32),
                base_offset=np.asarray(entry.base_offset, dtype=np.float32),
                surface_y=surface_y,
            )

        # Procedural terrain to cover side areas outside the road mesh.
        ground_key = "ground_terrain"
        ground_mesh = build_ground_plane_mesh(width=58.0, depth=130.0)
        self.renderer.register_mesh(ground_key, ground_mesh)
        self.assets[ground_key] = RuntimeAsset(
            entry=AssetEntry(
                key=ground_key,
                category="road",
                notes="Procedural ground plane for side terrain coverage.",
            ),
            mesh_key=ground_key,
            center_x=float((ground_mesh.bbox_min[0] + ground_mesh.bbox_max[0]) * 0.5),
            min_y=float(ground_mesh.bbox_min[1]),
            center_z=float((ground_mesh.bbox_min[2] + ground_mesh.bbox_max[2]) * 0.5),
            scale_vec=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            size_world=np.asarray(ground_mesh.extents, dtype=np.float32),
            length_world=float(max(ground_mesh.extents[0], ground_mesh.extents[2])),
            base_rotation_deg=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            base_offset=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            surface_y=0.0,
        )

    def _prepare_output_dirs(self) -> Dict[str, Path]:
        dirs = {
            "rgb": self.output_root / "images" / "rgb",
            "detection": self.output_root / "detection",
            "segment": self.output_root / "images" / "segment",
            "depth_npy": self.output_root / "depth" / "raw_npy",
            "depth_vis": self.output_root / "depth" / "vis",
            "mask_instance": self.output_root / "masks" / "instance",
            "mask_instance_vis": self.output_root / "masks" / "instance_vis",
            "yolo": self.output_root / "labels" / "yolo",
            "coco": self.output_root / "labels" / "coco",
            "meta_scene": self.output_root / "metadata" / "scenes",
            "meta_root": self.output_root / "metadata",
            "refs": self.output_root / "references",
            "videos": self.output_root / "videos",
        }
        for p in dirs.values():
            p.mkdir(parents=True, exist_ok=True)
        return dirs

    def _spawn_instance(
        self,
        inst_id: int,
        asset_key: str,
        position: np.ndarray,
        yaw_deg: float,
        extra_scale: float = 1.0,
        use_alpha_key: bool = False,
        wheel_spin: float = 0.0,
        wheel_slide: float = 0.0,
    ) -> Tuple[SceneInstance, FrameObject]:
        runtime = self.assets[asset_key]
        class_name = runtime.entry.category
        class_id = CLASS_TO_ID[class_name]
        if not use_alpha_key and class_name in {"person", "tree"}:
            use_alpha_key = True
        center = np.asarray([runtime.center_x, runtime.min_y, runtime.center_z], dtype=np.float32)
        model = _compose_model_matrix(
            position=position.astype(np.float32),
            yaw_deg=yaw_deg,
            local_center=center,
            base_rotation_deg=runtime.base_rotation_deg,
            base_scale=runtime.scale_vec,
            base_offset=runtime.base_offset,
            extra_scale=extra_scale,
        )

        scene_instance = SceneInstance(
            mesh_key=runtime.mesh_key,
            class_id=class_id,
            class_name=class_name,
            instance_id=inst_id,
            model_matrix=model,
            use_alpha_key=use_alpha_key,
            wheel_spin=float(wheel_spin),
            wheel_slide=float(wheel_slide),
        )
        obj_meta = FrameObject(
            instance_id=inst_id,
            class_id=class_id,
            class_name=class_name,
            asset_key=asset_key,
            position=position.astype(np.float32),
            yaw_deg=float(yaw_deg),
            extra_scale=float(extra_scale),
        )
        return scene_instance, obj_meta

    def _build_background_instances(self) -> List[SceneInstance]:
        runtime = self.assets["ground_terrain"]
        center = np.asarray([runtime.center_x, runtime.min_y, runtime.center_z], dtype=np.float32)
        model = _compose_model_matrix(
            position=np.asarray([0.0, -0.02, -24.0], dtype=np.float32),
            yaw_deg=0.0,
            local_center=center,
            base_rotation_deg=runtime.base_rotation_deg,
            base_scale=runtime.scale_vec,
            base_offset=runtime.base_offset,
            extra_scale=1.0,
        )
        return [
            SceneInstance(
                mesh_key=runtime.mesh_key,
                class_id=CLASS_TO_ID["road"],
                class_name="road",
                instance_id=0,
                model_matrix=model,
                use_alpha_key=False,
            )
        ]

    def _draw_detection_overlay(
        self,
        rgb_image: np.ndarray,
        annos: List[Dict],
        objects: Dict[int, FrameObject],
    ) -> np.ndarray:
        canvas = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        for anno in annos:
            inst_id = int(anno["instance_id"])
            if inst_id not in objects:
                continue
            obj = objects[inst_id]
            x0, y0, bw, bh = [int(v) for v in anno["bbox_xywh"]]
            x1 = x0 + bw - 1
            y1 = y0 + bh - 1
            color = _class_color_bgr(obj.class_id)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2, lineType=cv2.LINE_AA)

            label = f"{obj.class_name}#{inst_id}"
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            label_top = max(0, y0 - th - baseline - 4)
            cv2.rectangle(
                canvas,
                (x0, label_top),
                (x0 + tw + 8, label_top + th + baseline + 6),
                color,
                -1,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                label,
                (x0 + 4, label_top + th + 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                1,
                lineType=cv2.LINE_AA,
            )
        return canvas

    def _init_sequence_actors(self, rng: np.random.Generator) -> List[SequenceActor]:
        cfg = self.scene_cfg["scene"]
        z_min = float(cfg["z_spawn_min"])
        z_max = float(cfg["z_spawn_max"])
        neg_lanes, pos_lanes, neg_dir, pos_dir = self._vehicle_lane_groups()

        actors: List[SequenceActor] = []
        next_id = 1
        traffic_y = self.road_surface_y

        def add_actor(
            asset_key: str,
            position: np.ndarray,
            yaw_deg: float,
            extra_scale: float,
            speed: float = 0.0,
            lane_x: Optional[float] = None,
            lateral_amp: float = 0.0,
            lateral_freq: float = 0.0,
            bob_amp: float = 0.0,
            bob_freq: float = 0.0,
            wheel_radius: float = 0.0,
            dynamic: bool = False,
            use_alpha_key: bool = False,
            respawn_min_z: Optional[float] = None,
            respawn_max_z: Optional[float] = None,
            direction_z: Optional[float] = None,
        ) -> None:
            nonlocal next_id
            asset = self.assets[asset_key]
            class_name = asset.entry.category
            flow_dir = self.traffic_direction_sign if direction_z is None else float(np.sign(direction_z))
            if abs(flow_dir) < 1e-6:
                flow_dir = self.traffic_direction_sign

            default_respawn_min = float(z_min - 12.0)
            default_respawn_max = 8.0
            if flow_dir < 0.0:
                default_respawn_min = float(z_min - 16.0)
                default_respawn_max = float(z_max + 2.0)

            actors.append(
                SequenceActor(
                    instance_id=next_id,
                    asset_key=asset_key,
                    class_name=class_name,
                    position=position.astype(np.float32),
                    base_y=float(position[1]),
                    yaw_deg=float(yaw_deg),
                    extra_scale=float(extra_scale),
                    speed=float(speed),
                    lane_x=float(position[0] if lane_x is None else lane_x),
                    lateral_amp=float(lateral_amp),
                    lateral_freq=float(lateral_freq),
                    lateral_phase=float(rng.uniform(0.0, 2.0 * math.pi)),
                    bob_amp=float(bob_amp),
                    bob_freq=float(bob_freq),
                    bob_phase=float(rng.uniform(0.0, 2.0 * math.pi)),
                    wheel_radius=float(wheel_radius),
                    use_alpha_key=bool(use_alpha_key),
                    dynamic=bool(dynamic),
                    respawn_min_z=float(default_respawn_min if respawn_min_z is None else respawn_min_z),
                    respawn_max_z=float(default_respawn_max if respawn_max_z is None else respawn_max_z),
                    direction_z=float(flow_dir),
                )
            )
            next_id += 1

        # Static base assets.
        add_actor("road_street", np.asarray([0.0, 0.0, -24.0], dtype=np.float32), yaw_deg=0.0, extra_scale=1.0)
        add_actor(
            "building_city",
            np.asarray([-18.0, 0.0, -24.0], dtype=np.float32),
            yaw_deg=90.0,
            extra_scale=1.05,
        )
        add_actor(
            "building_city",
            np.asarray([18.5, 0.0, -24.0], dtype=np.float32),
            yaw_deg=-90.0,
            extra_scale=0.95,
        )

        tree_count = _randint_pair(rng, cfg["n_trees"])
        for _ in range(tree_count):
            side = -1.0 if rng.random() < 0.5 else 1.0
            x = side * float(rng.uniform(10.5, 14.5))
            z = float(rng.uniform(z_min + 8.0, -7.0))
            add_actor(
                "tree_tyro",
                np.asarray([x, 0.0, z], dtype=np.float32),
                yaw_deg=float(rng.uniform(0.0, 360.0)),
                extra_scale=float(rng.uniform(0.8, 1.25)),
            )

        car_total = _randint_pair(rng, cfg["n_cars"])
        car_dirs: List[float] = []
        if car_total > 0:
            if car_total == 1:
                car_dirs = [neg_dir]
            else:
                n_neg = car_total // 2
                n_pos = car_total - n_neg
                car_dirs = [neg_dir] * n_neg + [pos_dir] * n_pos
                rng.shuffle(car_dirs)
        for direction in car_dirs:
            lane_choices = neg_lanes if direction == neg_dir else pos_lanes
            lane = float(rng.choice(lane_choices))
            if direction > 0.0:
                z = float(rng.uniform(z_min + 2.0, z_max))
                respawn_min = float(z_min - 14.0)
                respawn_max = 8.0
            else:
                z = float(rng.uniform(z_max + 1.0, z_max + 14.0))
                respawn_min = float(z_min - 16.0)
                respawn_max = float(z_max + 3.0)
            asset = "car_audi_r8" if rng.random() < 0.7 else "car_mercedes"
            add_actor(
                asset,
                np.asarray([lane, traffic_y, z], dtype=np.float32),
                yaw_deg=self._yaw_from_direction(direction, jitter_deg=6.0, rng=rng),
                extra_scale=float(rng.uniform(0.93, 1.08)),
                speed=float(rng.uniform(7.5, 12.0)),
                lane_x=lane,
                lateral_amp=float(rng.uniform(0.05, 0.22)),
                lateral_freq=float(rng.uniform(0.8, 1.7)),
                wheel_radius=float(rng.uniform(0.30, 0.38)),
                dynamic=True,
                respawn_min_z=respawn_min,
                respawn_max_z=respawn_max,
                direction_z=direction,
            )

        truck_total = _randint_pair(rng, cfg["n_trucks"])
        truck_dirs: List[float] = []
        if truck_total > 0:
            if truck_total == 1:
                truck_dirs = [neg_dir if rng.random() < 0.5 else pos_dir]
            else:
                n_neg = truck_total // 2
                n_pos = truck_total - n_neg
                truck_dirs = [neg_dir] * n_neg + [pos_dir] * n_pos
                rng.shuffle(truck_dirs)
        for direction in truck_dirs:
            lane_choices = neg_lanes if direction == neg_dir else pos_lanes
            lane = float(rng.choice(lane_choices))
            if direction > 0.0:
                z = float(rng.uniform(z_min + 4.0, z_max - 1.0))
                respawn_min = float(z_min - 16.0)
                respawn_max = 9.0
            else:
                z = float(rng.uniform(z_max + 1.5, z_max + 11.0))
                respawn_min = float(z_min - 17.0)
                respawn_max = float(z_max + 2.5)
            add_actor(
                "truck_main",
                np.asarray([lane, traffic_y, z], dtype=np.float32),
                yaw_deg=self._yaw_from_direction(direction, jitter_deg=4.0, rng=rng),
                extra_scale=float(rng.uniform(0.95, 1.08)),
                speed=float(rng.uniform(5.5, 8.4)),
                lane_x=lane,
                lateral_amp=float(rng.uniform(0.03, 0.12)),
                lateral_freq=float(rng.uniform(0.6, 1.2)),
                wheel_radius=float(rng.uniform(0.42, 0.56)),
                dynamic=True,
                respawn_min_z=respawn_min,
                respawn_max_z=respawn_max,
                direction_z=direction,
            )

        person_total = _randint_pair(rng, cfg["n_persons"])
        person_z_samples = (
            rng.uniform(z_min + 10.0, z_max + 8.0, person_total)
            if self.traffic_direction_sign > 0.0
            else rng.uniform(z_max - 1.0, z_max + 6.0, person_total)
        )
        for z in person_z_samples.tolist():
            side = -1.0 if rng.random() < 0.5 else 1.0
            x = side * float(rng.uniform(5.8, 8.2))
            yaw = 180.0 if side < 0 else 0.0
            add_actor(
                "person_billboard",
                np.asarray([x, traffic_y, z], dtype=np.float32),
                yaw_deg=float(yaw),
                extra_scale=float(rng.uniform(0.92, 1.1)),
                speed=float(rng.uniform(0.8, 1.8)),
                lane_x=x,
                lateral_amp=float(rng.uniform(0.05, 0.18)),
                lateral_freq=float(rng.uniform(0.6, 1.5)),
                bob_amp=float(rng.uniform(0.02, 0.05)),
                bob_freq=float(rng.uniform(8.0, 12.0)),
                dynamic=True,
                use_alpha_key=True,
                respawn_min_z=z_min - 8.0,
                respawn_max_z=(6.0 if self.traffic_direction_sign > 0.0 else float(z_max + 5.0)),
                direction_z=self.traffic_direction_sign,
            )

        self._enforce_vehicle_lane_spacing(actors, gap_buffer=1.6)
        return actors

    def _advance_sequence_actors(
        self,
        actors: List[SequenceActor],
        rng: np.random.Generator,
        dt: float,
        sim_time: float,
    ) -> None:
        neg_lanes, pos_lanes, neg_dir, _ = self._vehicle_lane_groups()

        for actor in actors:
            if not actor.dynamic:
                continue

            prev_x = float(actor.position[0])
            prev_z = float(actor.position[2])

            actor.position[2] = np.float32(float(actor.position[2]) + actor.direction_z * actor.speed * dt)
            actor.position[0] = np.float32(
                actor.lane_x + actor.lateral_amp * math.sin(actor.lateral_freq * sim_time + actor.lateral_phase)
            )
            actor.position[1] = np.float32(
                actor.base_y + actor.bob_amp * math.sin(actor.bob_freq * sim_time + actor.bob_phase)
            )

            travel = math.sqrt((float(actor.position[2]) - prev_z) ** 2 + (float(actor.position[0]) - prev_x) ** 2)
            slip = abs(float(actor.position[0]) - prev_x)
            if actor.wheel_radius > 1e-4:
                circum = 2.0 * math.pi * actor.wheel_radius
                actor.wheel_spin = float((actor.wheel_spin + travel / circum) % 1.0)
                actor.wheel_slide = float((actor.wheel_slide + slip / circum) % 1.0)

            if actor.direction_z > 0.0:
                should_respawn = float(actor.position[2]) > actor.respawn_max_z
            else:
                should_respawn = float(actor.position[2]) < actor.respawn_min_z
            if not should_respawn:
                continue

            if actor.direction_z > 0.0:
                actor.position[2] = np.float32(actor.respawn_min_z - float(rng.uniform(0.0, 16.0)))
            else:
                actor.position[2] = np.float32(actor.respawn_max_z + float(rng.uniform(0.0, 10.0)))
            actor.lateral_phase = float(rng.uniform(0.0, 2.0 * math.pi))
            actor.bob_phase = float(rng.uniform(0.0, 2.0 * math.pi))
            actor.wheel_spin = float(rng.uniform(0.0, 1.0))
            actor.wheel_slide = float(rng.uniform(0.0, 1.0))

            if actor.class_name in {"car", "truck"}:
                lane_choices = neg_lanes if float(actor.direction_z) == float(neg_dir) else pos_lanes
                actor.lane_x = float(rng.choice(lane_choices))
                actor.speed = float(np.clip(actor.speed + rng.normal(0.0, 0.5), 3.5, 13.5))
                jitter = 4.0 if actor.class_name == "truck" else 10.0
                actor.yaw_deg = self._yaw_from_direction(actor.direction_z, jitter_deg=jitter, rng=rng)
            elif actor.class_name == "person":
                side = -1.0 if rng.random() < 0.5 else 1.0
                actor.lane_x = side * float(rng.uniform(5.8, 8.2))
                actor.yaw_deg = 180.0 if side < 0 else 0.0
                actor.speed = float(np.clip(actor.speed + rng.normal(0.0, 0.2), 0.6, 2.0))

        self._enforce_vehicle_lane_spacing(actors, gap_buffer=1.6)

    def _compose_sequence_scene(
        self,
        actors: List[SequenceActor],
    ) -> Tuple[List[SceneInstance], Dict[int, FrameObject]]:
        scene_instances: List[SceneInstance] = list(self._background_instances)
        objects: Dict[int, FrameObject] = {}
        for actor in actors:
            inst, meta = self._spawn_instance(
                inst_id=actor.instance_id,
                asset_key=actor.asset_key,
                position=actor.position,
                yaw_deg=actor.yaw_deg,
                extra_scale=actor.extra_scale,
                use_alpha_key=actor.use_alpha_key,
                wheel_spin=actor.wheel_spin,
                wheel_slide=actor.wheel_slide,
            )
            scene_instances.append(inst)
            objects[actor.instance_id] = meta
        return scene_instances, objects

    def _sample_frame_scene(
        self,
        rng: np.random.Generator,
    ) -> Tuple[List[SceneInstance], Dict[int, FrameObject]]:
        cfg = self.scene_cfg["scene"]
        z_min = float(cfg["z_spawn_min"])
        z_max = float(cfg["z_spawn_max"])
        neg_lanes, pos_lanes, neg_dir, pos_dir = self._vehicle_lane_groups()

        scene_instances: List[SceneInstance] = []
        objects: Dict[int, FrameObject] = {}
        next_id = 1

        def add(
            asset_key: str,
            position: np.ndarray,
            yaw_deg: float,
            extra_scale: float = 1.0,
            use_alpha_key: bool = False,
        ) -> None:
            nonlocal next_id
            inst, meta = self._spawn_instance(
                inst_id=next_id,
                asset_key=asset_key,
                position=position,
                yaw_deg=yaw_deg,
                extra_scale=extra_scale,
                use_alpha_key=use_alpha_key,
            )
            scene_instances.append(inst)
            objects[next_id] = meta
            next_id += 1

        def add_background(
            asset_key: str,
            position: np.ndarray,
            yaw_deg: float,
            extra_scale: float = 1.0,
        ) -> None:
            runtime = self.assets[asset_key]
            center = np.asarray([runtime.center_x, runtime.min_y, runtime.center_z], dtype=np.float32)
            model = _compose_model_matrix(
                position=position.astype(np.float32),
                yaw_deg=yaw_deg,
                local_center=center,
                base_rotation_deg=runtime.base_rotation_deg,
                base_scale=runtime.scale_vec,
                base_offset=runtime.base_offset,
                extra_scale=extra_scale,
            )
            scene_instances.append(
                SceneInstance(
                    mesh_key=runtime.mesh_key,
                    class_id=CLASS_TO_ID["road"],
                    class_name="road",
                    instance_id=0,
                    model_matrix=model,
                    use_alpha_key=False,
                )
            )

        # Static base scene
        add_background("ground_terrain", np.asarray([0.0, -0.02, -24.0], dtype=np.float32), yaw_deg=0.0)
        add("road_street", np.asarray([0.0, 0.0, -24.0], dtype=np.float32), yaw_deg=0.0)
        add("building_city", np.asarray([-18.0, 0.0, -24.0], dtype=np.float32), yaw_deg=90.0, extra_scale=1.05)
        add("building_city", np.asarray([18.5, 0.0, -24.0], dtype=np.float32), yaw_deg=-90.0, extra_scale=0.95)
        traffic_y = self.road_surface_y

        # Trees
        tree_count = _randint_pair(rng, cfg["n_trees"])
        for _ in range(tree_count):
            side = -1.0 if rng.random() < 0.5 else 1.0
            x = side * rng.uniform(10.5, 14.5)
            z = rng.uniform(z_min + 8.0, -7.0)
            yaw = rng.uniform(0.0, 360.0)
            scale = rng.uniform(0.8, 1.25)
            add("tree_tyro", np.asarray([x, 0.0, z], dtype=np.float32), yaw_deg=yaw, extra_scale=scale)

        # Cars
        car_total = _randint_pair(rng, cfg["n_cars"])
        car_dirs: List[float] = []
        if car_total > 0:
            if car_total == 1:
                car_dirs = [neg_dir]
            else:
                n_neg = car_total // 2
                n_pos = car_total - n_neg
                car_dirs = [neg_dir] * n_neg + [pos_dir] * n_pos
                rng.shuffle(car_dirs)
        for direction in car_dirs:
            lane_choices = neg_lanes if direction == neg_dir else pos_lanes
            lane = float(rng.choice(lane_choices))
            if direction > 0.0:
                z = float(rng.uniform(z_min + 2.0, z_max + 1.0))
            else:
                z = float(rng.uniform(z_max + 1.0, z_max + 14.0))
            asset = "car_audi_r8" if rng.random() < 0.7 else "car_mercedes"
            yaw = self._yaw_from_direction(direction, jitter_deg=6.0, rng=rng)
            add(asset, np.asarray([lane, traffic_y, z], dtype=np.float32), yaw_deg=yaw, extra_scale=rng.uniform(0.93, 1.08))

        # Trucks
        truck_total = _randint_pair(rng, cfg["n_trucks"])
        truck_dirs: List[float] = []
        if truck_total > 0:
            if truck_total == 1:
                truck_dirs = [neg_dir if rng.random() < 0.5 else pos_dir]
            else:
                n_neg = truck_total // 2
                n_pos = truck_total - n_neg
                truck_dirs = [neg_dir] * n_neg + [pos_dir] * n_pos
                rng.shuffle(truck_dirs)
        for direction in truck_dirs:
            lane_choices = neg_lanes if direction == neg_dir else pos_lanes
            lane = float(rng.choice(lane_choices))
            if direction > 0.0:
                z = float(rng.uniform(z_min + 4.0, z_max - 1.0))
            else:
                z = float(rng.uniform(z_max + 1.5, z_max + 11.0))
            yaw = self._yaw_from_direction(direction, jitter_deg=4.0, rng=rng)
            add("truck_main", np.asarray([lane, traffic_y, z], dtype=np.float32), yaw_deg=yaw, extra_scale=rng.uniform(0.95, 1.08))

        # Pedestrians
        person_total = _randint_pair(rng, cfg["n_persons"])
        for z in rng.uniform(z_min + 10.0, z_max + 8.0, person_total):
            side = -1.0 if rng.random() < 0.5 else 1.0
            x = side * rng.uniform(5.8, 8.2)
            yaw = 180.0 if side < 0 else 0.0
            add(
                "person_billboard",
                np.asarray([x, traffic_y, z], dtype=np.float32),
                yaw_deg=yaw,
                extra_scale=rng.uniform(0.92, 1.1),
                use_alpha_key=True,
            )

        return scene_instances, objects

    def _depth_to_visual(self, depth_linear: np.ndarray) -> np.ndarray:
        finite = np.isfinite(depth_linear)
        if not np.any(finite):
            return np.zeros((depth_linear.shape[0], depth_linear.shape[1]), dtype=np.uint8)
        d = depth_linear.copy()
        d[~finite] = np.nanmax(d[finite])
        d_min = float(np.nanpercentile(d, 1))
        d_max = float(np.nanpercentile(d, 99))
        if d_max <= d_min + 1e-6:
            d_max = d_min + 1.0
        norm = np.clip((d - d_min) / (d_max - d_min), 0.0, 1.0)
        return (255.0 * (1.0 - norm)).astype(np.uint8)

    def _clean_instance_map_fragments(
        self,
        instance_map: np.ndarray,
        objects: Optional[Dict[int, FrameObject]] = None,
        min_component_pixels: int = 28,
    ) -> np.ndarray:
        cleaned = instance_map.copy()
        strict_single_component_classes = {"car", "truck", "bike", "moto", "person"}
        relative_keep_ratio_by_class = {
            "building": 0.08,
            "tree": 0.08,
            "road": 0.08,
        }
        for inst_id in np.unique(cleaned):
            inst_id_int = int(inst_id)
            if inst_id_int <= 0:
                continue
            mask = (cleaned == inst_id_int).astype(np.uint8)
            total = int(mask.sum())
            if total <= 0:
                continue
            n_comp, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            if n_comp <= 2:
                continue

            largest_comp = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            largest_area = int(stats[largest_comp, cv2.CC_STAT_AREA])
            obj = objects.get(inst_id_int) if objects is not None else None
            class_name = obj.class_name if obj is not None else ""
            keep = {largest_comp}

            # Vehicle/person meshes frequently include detached helper planes.
            # Keep the dominant silhouette only to avoid "floating" artifacts.
            if class_name not in strict_single_component_classes:
                rel_ratio = float(relative_keep_ratio_by_class.get(class_name, 0.08))
                min_keep_area = max(int(min_component_pixels), int(round(largest_area * rel_ratio)))
                for comp_idx in range(1, n_comp):
                    if comp_idx == largest_comp:
                        continue
                    if int(stats[comp_idx, cv2.CC_STAT_AREA]) >= int(min_keep_area):
                        keep.add(comp_idx)

            keep_mask = np.isin(labels, np.asarray(sorted(keep), dtype=np.int32))
            cleaned[(cleaned == inst_id_int) & (~keep_mask)] = 0
        return cleaned

    def _extract_annotations(
        self,
        instance_map: np.ndarray,
        objects: Dict[int, FrameObject],
    ) -> Tuple[List[Dict], List[str]]:
        annos: List[Dict] = []
        yolo_lines: List[str] = []
        h, w = instance_map.shape

        for inst_id, obj in objects.items():
            mask = (instance_map == inst_id).astype(np.uint8)
            area = int(mask.sum())
            if area <= 0:
                continue
            ys, xs = np.where(mask > 0)
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            bw = int(x1 - x0 + 1)
            bh = int(y1 - y0 + 1)
            obj.bbox_xywh = [x0, y0, bw, bh]
            obj.visible_pixels = area

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            segmentation: List[List[float]] = []
            for cnt in contours:
                if cnt.shape[0] < 3:
                    continue
                poly = cnt.reshape(-1, 2).astype(float).reshape(-1).tolist()
                if len(poly) >= 6:
                    segmentation.append(poly)
            if not segmentation:
                segmentation = [
                    [float(x0), float(y0), float(x1), float(y0), float(x1), float(y1), float(x0), float(y1)]
                ]

            annos.append(
                {
                    "instance_id": int(inst_id),
                    "class_id": int(obj.class_id),
                    "bbox_xywh": [x0, y0, bw, bh],
                    "area": area,
                    "segmentation": segmentation,
                }
            )

            cx = (x0 + bw * 0.5) / float(w)
            cy = (y0 + bh * 0.5) / float(h)
            nw = bw / float(w)
            nh = bh / float(h)
            yolo_lines.append(f"{obj.class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        return annos, yolo_lines

    def generate(self, frames: int, seed: int = 252) -> Dict[str, object]:
        dirs = self._prepare_output_dirs()
        export_catalog_json(self.sample_root, self.catalog, dirs["meta_root"] / "asset_catalog.json")

        refs = collect_reference_images(self.sample_root, self.catalog)
        for src in refs:
            dst = dirs["refs"] / src.name
            if not dst.exists():
                shutil.copy2(src, dst)

        coco_images: List[Dict] = []
        coco_annotations: List[Dict] = []
        coco_categories = [{"id": i + 1, "name": cls} for i, cls in enumerate(CLASS_NAMES)]
        coco_ann_id = 1

        rng = np.random.default_rng(seed)
        dt = 1.0 / float(self.fps)
        sequence_actors: Optional[List[SequenceActor]] = None
        if self.scene_mode == "sequence":
            sequence_actors = self._init_sequence_actors(rng)

        rgb_writer: Optional[cv2.VideoWriter] = None
        det_writer: Optional[cv2.VideoWriter] = None
        seg_writer: Optional[cv2.VideoWriter] = None
        panel_writer: Optional[cv2.VideoWriter] = None
        video_paths: Dict[str, str] = {}

        if self.write_video and frames > 0:
            rgb_writer, rgb_path = _create_video_writer(dirs["videos"] / "rgb.mp4", self.width, self.height, self.fps)
            det_writer, det_path = _create_video_writer(
                dirs["videos"] / "detection.mp4",
                self.width,
                self.height,
                self.fps,
            )
            seg_writer, seg_path = _create_video_writer(
                dirs["videos"] / "segment.mp4",
                self.width,
                self.height,
                self.fps,
            )
            panel_writer, panel_path = _create_video_writer(
                dirs["videos"] / "triplet_rgb_segment_detection.mp4",
                self.width * 3,
                self.height,
                self.fps,
            )
            video_paths = {
                "rgb_video": str(rgb_path.relative_to(self.output_root)).replace("\\", "/"),
                "detection_video": str(det_path.relative_to(self.output_root)).replace("\\", "/"),
                "segment_video": str(seg_path.relative_to(self.output_root)).replace("\\", "/"),
                "triplet_video": str(panel_path.relative_to(self.output_root)).replace("\\", "/"),
            }

        try:
            for frame_idx in range(frames):
                frame_id = frame_idx + 1
                stem = f"{frame_id:06d}"

                if self.scene_mode == "sequence":
                    assert sequence_actors is not None
                    sim_time = frame_idx * dt
                    if frame_idx > 0:
                        self._advance_sequence_actors(sequence_actors, rng, dt=dt, sim_time=sim_time)
                    scene_instances, frame_objects = self._compose_sequence_scene(sequence_actors)
                else:
                    scene_instances, frame_objects = self._sample_frame_scene(rng)

                render = self.renderer.render(self.camera, self.light, scene_instances)
                instance_map = self._clean_instance_map_fragments(render.instance_map, objects=frame_objects)
                rgb_out = self._choose_rgb_output(stem, render.rgb)
                segment_rgb = _instance_map_to_color(instance_map)
                segment_bgr = cv2.cvtColor(segment_rgb, cv2.COLOR_RGB2BGR)

                annos, yolo_lines = self._extract_annotations(instance_map, frame_objects)
                detection_bgr = self._draw_detection_overlay(rgb_out, annos, frame_objects)

                rgb_path = dirs["rgb"] / f"{stem}.png"
                detection_path = dirs["detection"] / f"{stem}.png"
                segment_path = dirs["segment"] / f"{stem}.png"
                depth_npy_path = dirs["depth_npy"] / f"{stem}.npy"
                depth_vis_path = dirs["depth_vis"] / f"{stem}.png"
                inst_mask_path = dirs["mask_instance"] / f"{stem}.png"
                yolo_path = dirs["yolo"] / f"{stem}.txt"
                scene_meta_path = dirs["meta_scene"] / f"{stem}.json"

                rgb_bgr = cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR)
                _write_image_unicode(rgb_path, rgb_bgr)
                _write_image_unicode(detection_path, detection_bgr)
                _write_image_unicode(segment_path, segment_bgr)
                _write_npy_unicode(depth_npy_path, render.depth_linear.astype(np.float32))
                _write_image_unicode(depth_vis_path, self._depth_to_visual(render.depth_linear))
                _write_image_unicode(inst_mask_path, instance_map.astype(np.uint16))
                _write_image_unicode(dirs["mask_instance_vis"] / f"{stem}.png", segment_bgr)

                yolo_path.write_text("\n".join(yolo_lines), encoding="utf-8")

                if rgb_writer is not None:
                    rgb_writer.write(rgb_bgr)
                if det_writer is not None:
                    det_writer.write(detection_bgr)
                if seg_writer is not None:
                    seg_writer.write(segment_bgr)
                if panel_writer is not None:
                    panel_writer.write(np.hstack([rgb_bgr, segment_bgr, detection_bgr]))

                coco_images.append(
                    {
                        "id": frame_id,
                        "file_name": f"images/rgb/{stem}.png",
                        "width": self.width,
                        "height": self.height,
                    }
                )

                for anno in annos:
                    coco_annotations.append(
                        {
                            "id": coco_ann_id,
                            "image_id": frame_id,
                            "category_id": int(anno["class_id"]) + 1,
                            "bbox": [float(v) for v in anno["bbox_xywh"]],
                            "area": float(anno["area"]),
                            "segmentation": anno["segmentation"],
                            "iscrowd": 0,
                        }
                    )
                    coco_ann_id += 1

                scene_meta_payload = {
                    "frame_id": frame_id,
                    "scene_mode": self.scene_mode,
                    "rgb_source": self.rgb_source,
                    "camera_profile": self.camera_view,
                    "time_of_day": self.time_of_day,
                    "traffic_flow": self.traffic_flow,
                    "rgb_path": str(rgb_path.relative_to(self.output_root)).replace("\\", "/"),
                    "detection_path": str(detection_path.relative_to(self.output_root)).replace("\\", "/"),
                    "segment_vis_path": str(segment_path.relative_to(self.output_root)).replace("\\", "/"),
                    "depth_path": str(depth_npy_path.relative_to(self.output_root)).replace("\\", "/"),
                    "instance_mask_path": str(inst_mask_path.relative_to(self.output_root)).replace("\\", "/"),
                    "camera": {
                        "eye": self.camera.eye.tolist(),
                        "target": self.camera.target.tolist(),
                        "up": self.camera.up.tolist(),
                        "fovy": self.camera.fovy,
                        "near": self.camera.near,
                        "far": self.camera.far,
                    },
                    "lighting": {
                        "sun_direction": self.light.sun_direction.tolist(),
                        "sun_color": self.light.sun_color.tolist(),
                        "sun_intensity": float(self.light.sun_intensity),
                        "fill_position": self.light.fill_position.tolist(),
                        "fill_color": self.light.fill_color.tolist(),
                        "fill_intensity": float(self.light.fill_intensity),
                        "street_light_color": self.light.street_light_color.tolist(),
                        "street_light_intensity": float(self.light.street_light_intensity),
                        "street_light_count": int(np.asarray(self.light.street_light_positions).reshape(-1, 3).shape[0]),
                        "sky_color": self.light.sky_color.tolist(),
                        "ambient_strength": float(self.light.ambient_strength),
                        "shadow_enabled": bool(self.light.shadow_enabled),
                        "shadow_strength": float(self.light.shadow_strength),
                    },
                    "objects": [
                        {
                            "instance_id": obj.instance_id,
                            "class_id": obj.class_id,
                            "class_name": obj.class_name,
                            "asset_key": obj.asset_key,
                            "position": [float(v) for v in obj.position.tolist()],
                            "yaw_deg": float(obj.yaw_deg),
                            "extra_scale": float(obj.extra_scale),
                            "bbox_xywh": obj.bbox_xywh,
                            "visible_pixels": int(obj.visible_pixels),
                        }
                        for obj in frame_objects.values()
                    ],
                }
                scene_meta_path.write_text(
                    json.dumps(scene_meta_payload, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
        finally:
            for writer in (rgb_writer, det_writer, seg_writer, panel_writer):
                if writer is not None:
                    writer.release()

        coco_payload = {
            "info": {
                "description": "BTL2 synthetic traffic dataset",
                "version": "1.0",
                "year": 2026,
            },
            "licenses": [],
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": coco_categories,
        }
        (dirs["coco"] / "instances_train.json").write_text(
            json.dumps(coco_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        summary: Dict[str, object] = {
            "frames": frames,
            "images": len(coco_images),
            "annotations": len(coco_annotations),
            "classes": len(CLASS_NAMES),
            "rgb_source": self.rgb_source,
            "gaussian_rgb_hits": int(self._gaussian_rgb_hits),
            "gaussian_rgb_misses": int(self._gaussian_rgb_misses),
            "scene_mode": self.scene_mode,
            "camera_profile": self.camera_view,
            "time_of_day": self.time_of_day,
            "traffic_flow": self.traffic_flow,
            "sun_intensity": float(self.light.sun_intensity),
            "fill_intensity": float(self.light.fill_intensity),
            "street_light_intensity": float(self.light.street_light_intensity),
            "street_light_count": int(np.asarray(self.light.street_light_positions).reshape(-1, 3).shape[0]),
            "shadow_enabled": bool(self.light.shadow_enabled),
            "shadow_strength": float(self.light.shadow_strength),
            "video_mode": ("enabled" if self.write_video else "disabled"),
        }
        for key, rel_path in video_paths.items():
            summary[key] = rel_path
        return summary