from __future__ import annotations

import argparse
import sys
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
SAMPLE_ROOT = THIS_FILE.parent.parent.resolve()
if str(SAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_ROOT))

from btl2_sys.core.reconstruct_from_2d import (  # noqa: E402
    Intrinsics,
    ReconstructionConfig,
    TwoDToThreeDGenerator,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Advanced 2D-to-3D data generation: estimate depth from frames, "
            "reconstruct per-frame/fused point clouds, and export metadata."
        )
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        required=True,
        help="Input directory containing 2D frames (png/jpg/jpeg/bmp/webp).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SAMPLE_ROOT / "btl2_sys" / "output_2d_to_3d",
        help="Output directory for generated 3D data.",
    )
    parser.add_argument(
        "--camera-meta-dir",
        type=Path,
        default=None,
        help="Optional per-frame camera metadata directory (000001.json, ...).",
    )
    parser.add_argument(
        "--depth-npy-dir",
        type=Path,
        default=None,
        help="Optional depth npy directory. If provided, depth estimation is skipped for matching frames.",
    )

    parser.add_argument("--near", type=float, default=1.0, help="Near depth in meters.")
    parser.add_argument("--far", type=float, default=60.0, help="Far depth in meters.")
    parser.add_argument("--sample-stride", type=int, default=2, help="Pixel stride for point sampling.")
    parser.add_argument(
        "--temporal-smoothing",
        type=float,
        default=0.35,
        help="Temporal smoothing factor in [0,1] using optical-flow warping.",
    )
    parser.add_argument(
        "--max-points-per-frame",
        type=int,
        default=180000,
        help="Maximum sampled points per frame.",
    )

    parser.add_argument("--fx", type=float, default=0.0, help="Camera fx. Use 0 to auto-infer from FOV.")
    parser.add_argument("--fy", type=float, default=0.0, help="Camera fy. Use 0 to auto-infer from FOV.")
    parser.add_argument("--cx", type=float, default=0.0, help="Camera cx. Use 0 to set width/2.")
    parser.add_argument("--cy", type=float, default=0.0, help="Camera cy. Use 0 to set height/2.")

    return parser


def main() -> int:
    args = _build_parser().parse_args()

    if not args.frames_dir.exists():
        raise FileNotFoundError(f"frames-dir does not exist: {args.frames_dir}")

    cfg = ReconstructionConfig(
        near=float(args.near),
        far=float(args.far),
        sample_stride=max(1, int(args.sample_stride)),
        temporal_smoothing=float(args.temporal_smoothing),
        max_points_per_frame=max(1, int(args.max_points_per_frame)),
    )

    intrinsics = None
    if args.fx > 0.0 and args.fy > 0.0:
        intrinsics = Intrinsics(
            fx=float(args.fx),
            fy=float(args.fy),
            cx=float(args.cx),
            cy=float(args.cy),
        )

    generator = TwoDToThreeDGenerator(
        frame_dir=args.frames_dir,
        output_dir=args.output,
        intrinsics=intrinsics,
        cfg=cfg,
        camera_meta_dir=args.camera_meta_dir,
        depth_npy_dir=args.depth_npy_dir,
    )

    summary = generator.run()

    print("[2D->3D] Generation complete.")
    print(f"  frames      : {summary.frames}")
    print(f"  points_total: {summary.points_total}")
    print(f"  depth_source: {summary.depth_source}")
    print(f"  fused_ply   : {summary.fused_ply}")
    print(f"  preview_perspective: {summary.preview_perspective_png}")
    print(f"  preview_top        : {summary.preview_top_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
