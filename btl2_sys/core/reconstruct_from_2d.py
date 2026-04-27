from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class ReconstructionConfig:
    near: float = 1.0
    far: float = 60.0
    sample_stride: int = 2
    temporal_smoothing: float = 0.35
    max_points_per_frame: int = 180_000


@dataclass
class ReconstructionSummary:
    frames: int
    points_total: int
    fused_ply: str
    depth_source: str
    preview_perspective_png: str
    preview_top_png: str


def _read_image_unicode_rgb(path: Path) -> np.ndarray:
    raw = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Unable to read image: {path}")

    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _write_image_unicode(path: Path, image: np.ndarray) -> None:
    suffix = path.suffix.lower()
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise RuntimeError(f"cv2.imencode failed for {path}")
    path.write_bytes(encoded.tobytes())


def _write_npy_unicode(path: Path, array: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(array), allow_pickle=False)
    path.write_bytes(buffer.getvalue())


def _depth_to_vis(depth: np.ndarray, near: float, far: float) -> np.ndarray:
    d = np.clip(depth, near, far)
    norm = (d - near) / max(far - near, 1e-6)
    vis = (255.0 * (1.0 - norm)).astype(np.uint8)
    return cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)


def _normalize01(values: np.ndarray) -> np.ndarray:
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax <= vmin + 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - vmin) / (vmax - vmin)).astype(np.float32)


def _estimate_depth_heuristic(rgb: np.ndarray, near: float, far: float) -> np.ndarray:
    h, w = rgb.shape[:2]
    rgb_f = rgb.astype(np.float32) / 255.0

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gray_blur = cv2.GaussianBlur(gray, (0, 0), 1.2)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32) / 255.0
    sat = hsv[:, :, 1]

    lap = cv2.Laplacian(gray_blur, cv2.CV_32F, ksize=3)
    edge_strength = _normalize01(np.abs(lap))
    local_contrast = _normalize01(np.abs(gray - gray_blur))

    y_grid = np.linspace(0.0, 1.0, h, dtype=np.float32).reshape(h, 1)
    near_prior = np.repeat(y_grid, w, axis=1)

    colorfulness = _normalize01(np.std(rgb_f, axis=2))

    # Relative closeness score in [0, 1].
    near_score = (
        0.56 * near_prior
        + 0.18 * local_contrast
        + 0.14 * sat
        + 0.12 * colorfulness
        - 0.08 * edge_strength
    )
    near_score = np.clip(near_score, 0.0, 1.0)
    near_score = cv2.bilateralFilter(near_score.astype(np.float32), 5, 0.08, 2.2)

    depth = near + (1.0 - near_score) * (far - near)
    return depth.astype(np.float32)


def _smooth_depth_temporal(
    prev_gray: Optional[np.ndarray],
    prev_depth: Optional[np.ndarray],
    curr_gray: np.ndarray,
    curr_depth: np.ndarray,
    alpha: float,
) -> np.ndarray:
    if prev_gray is None or prev_depth is None:
        return curr_depth

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=17,
        iterations=2,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )

    h, w = curr_depth.shape
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = grid_x - flow[:, :, 0]
    map_y = grid_y - flow[:, :, 1]

    warped_prev = cv2.remap(
        prev_depth,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    blended = (1.0 - alpha) * warped_prev + alpha * curr_depth
    return blended.astype(np.float32)


def _default_intrinsics(width: int, height: int, fov_deg: float = 60.0) -> Intrinsics:
    fx = (width * 0.5) / np.tan(np.deg2rad(fov_deg * 0.5))
    fy = fx
    cx = width * 0.5
    cy = height * 0.5
    return Intrinsics(fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy))


def _backproject_points(
    depth: np.ndarray,
    rgb: np.ndarray,
    intr: Intrinsics,
    sample_stride: int,
    max_points: int,
) -> np.ndarray:
    if sample_stride > 1:
        depth_s = depth[::sample_stride, ::sample_stride]
        rgb_s = rgb[::sample_stride, ::sample_stride]
        yy, xx = np.meshgrid(
            np.arange(depth.shape[0], dtype=np.float32)[::sample_stride],
            np.arange(depth.shape[1], dtype=np.float32)[::sample_stride],
            indexing="ij",
        )
    else:
        depth_s = depth
        rgb_s = rgb
        yy, xx = np.meshgrid(
            np.arange(depth.shape[0], dtype=np.float32),
            np.arange(depth.shape[1], dtype=np.float32),
            indexing="ij",
        )

    z = depth_s.reshape(-1)
    x = ((xx.reshape(-1) - intr.cx) * z) / intr.fx
    y = ((yy.reshape(-1) - intr.cy) * z) / intr.fy

    points = np.stack([x, y, z], axis=1).astype(np.float32)
    colors = rgb_s.reshape(-1, 3).astype(np.uint8)

    if points.shape[0] > max_points:
        idx = np.linspace(0, points.shape[0] - 1, num=max_points, dtype=np.int64)
        points = points[idx]
        colors = colors[idx]

    return np.hstack([points, colors.astype(np.float32)])


def _camera_to_world_matrix(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward = forward / max(np.linalg.norm(forward), 1e-8)
    right = np.cross(forward, up)
    right = right / max(np.linalg.norm(right), 1e-8)
    up2 = np.cross(right, forward)

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = up2
    c2w[:3, 2] = forward
    c2w[:3, 3] = eye
    return c2w


def _transform_points(points_xyzrgb: np.ndarray, transform_4x4: np.ndarray) -> np.ndarray:
    xyz = points_xyzrgb[:, :3]
    rgb = points_xyzrgb[:, 3:6]
    ones = np.ones((xyz.shape[0], 1), dtype=np.float32)
    hom = np.hstack([xyz, ones])
    xyz_world = (hom @ transform_4x4.T)[:, :3]
    return np.hstack([xyz_world, rgb])


def _write_ply_xyzrgb(path: Path, points_xyzrgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    n = int(points_xyzrgb.shape[0])
    header = "\n".join(
        [
            "ply",
            "format ascii 1.0",
            f"element vertex {n}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header",
        ]
    )

    with path.open("w", encoding="utf-8") as f:
        f.write(header + "\n")
        for row in points_xyzrgb:
            x, y, z, r, g, b = row
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    z_axis = eye - target
    z_axis = z_axis / max(np.linalg.norm(z_axis), 1e-8)
    x_axis = np.cross(up, z_axis)
    x_axis = x_axis / max(np.linalg.norm(x_axis), 1e-8)
    y_axis = np.cross(z_axis, x_axis)

    view = np.eye(4, dtype=np.float32)
    view[0, :3] = x_axis
    view[1, :3] = y_axis
    view[2, :3] = z_axis
    view[0, 3] = -float(np.dot(x_axis, eye))
    view[1, 3] = -float(np.dot(y_axis, eye))
    view[2, 3] = -float(np.dot(z_axis, eye))
    return view


def _render_pointcloud_preview(
    points_xyzrgb: np.ndarray,
    out_path: Path,
    width: int,
    height: int,
    eye: np.ndarray,
    target: np.ndarray,
    up: np.ndarray,
    fov_deg: float = 58.0,
) -> None:
    canvas = np.full((height, width, 3), 242, dtype=np.uint8)
    if points_xyzrgb.shape[0] == 0:
        _write_image_unicode(out_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        return

    xyz = points_xyzrgb[:, :3].astype(np.float32)
    rgb = np.clip(points_xyzrgb[:, 3:6], 0.0, 255.0).astype(np.uint8)

    view = _look_at(eye=eye, target=target, up=up)
    hom = np.hstack([xyz, np.ones((xyz.shape[0], 1), dtype=np.float32)])
    cam = (hom @ view.T)[:, :3]

    # Camera looks toward -Z in this view convention.
    valid = cam[:, 2] < -1e-4
    if not np.any(valid):
        _write_image_unicode(out_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        return

    cam = cam[valid]
    rgb = rgb[valid]

    f = (width * 0.5) / max(np.tan(np.deg2rad(fov_deg * 0.5)), 1e-6)
    u = (cam[:, 0] * f) / (-cam[:, 2]) + (width * 0.5)
    v = (cam[:, 1] * f) / (-cam[:, 2]) + (height * 0.5)

    ui = np.rint(u).astype(np.int32)
    vi = np.rint(v).astype(np.int32)

    in_view = (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    if not np.any(in_view):
        _write_image_unicode(out_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        return

    ui = ui[in_view]
    vi = vi[in_view]
    zz = cam[in_view, 2]
    rgb = rgb[in_view]

    # Painter approach by depth: far to near then near overwrite.
    order = np.argsort(zz)
    ui = ui[order]
    vi = vi[order]
    rgb = rgb[order]

    canvas[vi, ui] = rgb
    # Small dilation to make sparse points easier to see.
    canvas = cv2.dilate(canvas, np.ones((2, 2), dtype=np.uint8), iterations=1)
    _write_image_unicode(out_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def _write_preview_images(points_xyzrgb: np.ndarray, out_dir: Path) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    perspective_path = out_dir / "fused_perspective.png"
    top_path = out_dir / "fused_top.png"

    if points_xyzrgb.shape[0] == 0:
        blank = np.zeros((720, 1280, 3), dtype=np.uint8)
        _write_image_unicode(perspective_path, blank)
        _write_image_unicode(top_path, blank)
        return {
            "perspective": str(perspective_path),
            "top": str(top_path),
        }

    xyz = points_xyzrgb[:, :3]
    center = np.median(xyz, axis=0).astype(np.float32)
    span = np.ptp(xyz, axis=0)
    radius = float(max(np.linalg.norm(span) * 0.38, 5.0))

    # Isometric-like perspective preview.
    eye_p = center + np.asarray([radius, radius * 0.62, radius], dtype=np.float32)
    _render_pointcloud_preview(
        points_xyzrgb=points_xyzrgb,
        out_path=perspective_path,
        width=1280,
        height=720,
        eye=eye_p,
        target=center,
        up=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        fov_deg=56.0,
    )

    # Top-down preview.
    eye_t = center + np.asarray([0.0, max(radius * 1.7, 8.0), 0.001], dtype=np.float32)
    _render_pointcloud_preview(
        points_xyzrgb=points_xyzrgb,
        out_path=top_path,
        width=1280,
        height=720,
        eye=eye_t,
        target=center,
        up=np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
        fov_deg=52.0,
    )

    return {
        "perspective": str(perspective_path),
        "top": str(top_path),
    }


class TwoDToThreeDGenerator:
    def __init__(
        self,
        frame_dir: Path,
        output_dir: Path,
        intrinsics: Optional[Intrinsics],
        cfg: ReconstructionConfig,
        camera_meta_dir: Optional[Path] = None,
        depth_npy_dir: Optional[Path] = None,
    ) -> None:
        self.frame_dir = frame_dir.resolve()
        self.output_dir = output_dir.resolve()
        self.cfg = cfg
        self.camera_meta_dir = camera_meta_dir.resolve() if camera_meta_dir is not None else None
        self.depth_npy_dir = depth_npy_dir.resolve() if depth_npy_dir is not None else None

        self.frames = sorted([p for p in self.frame_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file()])
        if not self.frames:
            raise FileNotFoundError(f"No image frames found in: {self.frame_dir}")

        first = _read_image_unicode_rgb(self.frames[0])
        self.height, self.width = first.shape[:2]
        self.intr = intrinsics if intrinsics is not None else _default_intrinsics(self.width, self.height)

    def _load_depth_if_available(self, stem: str) -> Optional[np.ndarray]:
        if self.depth_npy_dir is None:
            return None
        p = self.depth_npy_dir / f"{stem}.npy"
        if not p.exists():
            return None
        depth = np.load(p).astype(np.float32)
        if depth.shape != (self.height, self.width):
            depth = cv2.resize(depth, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        return depth

    def _camera_transform_for_frame(self, stem: str, frame_idx: int) -> np.ndarray:
        if self.camera_meta_dir is not None:
            meta_path = self.camera_meta_dir / f"{stem}.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                cam = meta.get("camera", {})
                eye = np.asarray(cam.get("eye", [0.0, 1.6, 0.0]), dtype=np.float32)
                target = np.asarray(cam.get("target", [0.0, 1.6, 1.0]), dtype=np.float32)
                up = np.asarray(cam.get("up", [0.0, 1.0, 0.0]), dtype=np.float32)
                return _camera_to_world_matrix(eye=eye, target=target, up=up)

        # Fallback: assume slow forward camera motion.
        t = np.eye(4, dtype=np.float32)
        t[2, 3] = -0.08 * float(frame_idx)
        t[1, 3] = 1.6
        return t

    def run(self) -> ReconstructionSummary:
        dirs: Dict[str, Path] = {
            "depth_npy": self.output_dir / "depth_est" / "raw_npy",
            "depth_vis": self.output_dir / "depth_est" / "vis",
            "pc_frame_ply": self.output_dir / "pointcloud" / "frame_ply",
            "pc_frame_npy": self.output_dir / "pointcloud" / "frame_npy",
            "pc_fused": self.output_dir / "pointcloud" / "fused",
            "meta": self.output_dir / "metadata",
        }
        for p in dirs.values():
            p.mkdir(parents=True, exist_ok=True)

        prev_gray: Optional[np.ndarray] = None
        prev_depth: Optional[np.ndarray] = None

        fused_points: List[np.ndarray] = []
        point_total = 0
        depth_source = "estimated_heuristic"

        for idx, frame_path in enumerate(self.frames):
            stem = frame_path.stem
            rgb = _read_image_unicode_rgb(frame_path)
            if rgb.shape[:2] != (self.height, self.width):
                rgb = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

            depth = self._load_depth_if_available(stem)
            if depth is not None:
                depth_source = "provided_depth_npy"
            else:
                depth = _estimate_depth_heuristic(rgb, near=self.cfg.near, far=self.cfg.far)

            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            depth = _smooth_depth_temporal(
                prev_gray=prev_gray,
                prev_depth=prev_depth,
                curr_gray=gray,
                curr_depth=depth,
                alpha=float(np.clip(self.cfg.temporal_smoothing, 0.0, 1.0)),
            )

            prev_gray = gray
            prev_depth = depth

            depth_path = dirs["depth_npy"] / f"{stem}.npy"
            _write_npy_unicode(depth_path, depth.astype(np.float32))

            depth_vis = _depth_to_vis(depth, near=self.cfg.near, far=self.cfg.far)
            _write_image_unicode(dirs["depth_vis"] / f"{stem}.png", depth_vis)

            frame_points = _backproject_points(
                depth=depth,
                rgb=rgb,
                intr=self.intr,
                sample_stride=max(1, int(self.cfg.sample_stride)),
                max_points=max(1, int(self.cfg.max_points_per_frame)),
            )

            c2w = self._camera_transform_for_frame(stem=stem, frame_idx=idx)
            frame_points_world = _transform_points(frame_points, c2w)
            fused_points.append(frame_points_world)
            point_total += int(frame_points_world.shape[0])

            _write_npy_unicode(dirs["pc_frame_npy"] / f"{stem}.npy", frame_points_world.astype(np.float32))
            _write_ply_xyzrgb(dirs["pc_frame_ply"] / f"{stem}.ply", frame_points_world)

        fused = np.concatenate(fused_points, axis=0) if fused_points else np.zeros((0, 6), dtype=np.float32)
        fused_ply_path = dirs["pc_fused"] / "fused_pointcloud.ply"
        _write_ply_xyzrgb(fused_ply_path, fused)
        preview_paths = _write_preview_images(fused, self.output_dir / "preview_3d")

        summary = ReconstructionSummary(
            frames=len(self.frames),
            points_total=int(point_total),
            fused_ply=str(fused_ply_path),
            depth_source=depth_source,
            preview_perspective_png=preview_paths["perspective"],
            preview_top_png=preview_paths["top"],
        )

        (dirs["meta"] / "reconstruction_summary.json").write_text(
            json.dumps(
                {
                    "frames": summary.frames,
                    "points_total": summary.points_total,
                    "fused_ply": summary.fused_ply,
                    "depth_source": summary.depth_source,
                    "preview_perspective_png": summary.preview_perspective_png,
                    "preview_top_png": summary.preview_top_png,
                    "intrinsics": {
                        "fx": self.intr.fx,
                        "fy": self.intr.fy,
                        "cx": self.intr.cx,
                        "cy": self.intr.cy,
                    },
                    "config": {
                        "near": self.cfg.near,
                        "far": self.cfg.far,
                        "sample_stride": self.cfg.sample_stride,
                        "temporal_smoothing": self.cfg.temporal_smoothing,
                        "max_points_per_frame": self.cfg.max_points_per_frame,
                    },
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        return summary
