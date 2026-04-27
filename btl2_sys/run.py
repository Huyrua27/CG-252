from __future__ import annotations

import argparse
import sys
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
SAMPLE_ROOT = THIS_FILE.parent.parent.resolve()
if str(SAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_ROOT))

from btl2_sys.core.archive_tools import extract_archives
from btl2_sys.core.dataset import BTL2DatasetGenerator


REQUIRED_ARCHIVES = [
    "xihu59td0bnk-AudiR8Spyder_2017.rar",
    "32-mercedes-benz-gls-580-2020.zip",
    "64-truck.zip",
    "54-mountain_bike.zip",
    "22-moto_simple.zip",
    "1f9jtr180dxk-Tree1ByTyroSmith.zip",
    "80-street.zip",
    "br0e6h1jamf4-building2.rar",
    "55-rp_nathan_animated_003_walking_fbx.zip",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BTL2 synthetic traffic dataset generator (RGB + depth + instance + COCO/YOLO)."
    )
    parser.add_argument("--frames", type=int, default=40, help="Number of frames to generate.")
    parser.add_argument("--width", type=int, default=1280, help="Render width.")
    parser.add_argument("--height", type=int, default=720, help="Render height.")
    parser.add_argument("--seed", type=int, default=252, help="Random seed.")
    parser.add_argument(
        "--output",
        type=Path,
        default=SAMPLE_ROOT / "btl2_sys" / "output",
        help="Output directory.",
    )
    parser.add_argument(
        "--scene-config",
        type=Path,
        default=SAMPLE_ROOT / "btl2_sys" / "config" / "default_scene.json",
        help="Scene configuration JSON path.",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Extract required archives from object/ before generation.",
    )
    parser.add_argument(
        "--max-faces",
        type=int,
        default=0,
        help="Maximum triangle count per mesh. Use 0 for full quality (no decimation).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=24,
        help="Video FPS for sequence rendering.",
    )
    parser.add_argument(
        "--scene-mode",
        type=str,
        choices=["sequence", "random"],
        default="sequence",
        help="Scene sampling mode. 'sequence' creates temporally coherent motion for video.",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Disable video export and only save per-frame images/labels.",
    )
    parser.add_argument(
        "--rgb-source",
        type=str,
        choices=["mesh", "gaussian_hybrid"],
        default="mesh",
        help="RGB source mode. 'mesh' uses OpenGL raster RGB. 'gaussian_hybrid' uses pre-rendered Gaussian RGB when available and keeps GT from mesh.",
    )
    parser.add_argument(
        "--gaussian-rgb-dir",
        type=Path,
        default=None,
        help="Directory with Gaussian RGB frames named 000001.png, 000002.png, ... Used when --rgb-source gaussian_hybrid.",
    )
    parser.add_argument(
        "--gaussian-blend",
        type=float,
        default=1.0,
        help="Blend factor in [0,1] when Gaussian RGB exists. 1.0 means full Gaussian RGB.",
    )
    parser.add_argument(
        "--time-of-day",
        type=str,
        choices=["day", "sunset", "night"],
        default="day",
        help="Lighting preset to simulate daytime conditions.",
    )
    parser.add_argument(
        "--camera-view",
        type=str,
        choices=["forward", "front", "left", "right", "rear", "back", "bird"],
        default="forward",
        help="Camera direction profile.",
    )
    parser.add_argument(
        "--sun-azimuth",
        type=float,
        default=None,
        help="Optional sun azimuth angle (degrees) to override preset/config.",
    )
    parser.add_argument(
        "--sun-elevation",
        type=float,
        default=None,
        help="Optional sun elevation angle (degrees) to override preset/config.",
    )
    parser.add_argument(
        "--sun-intensity",
        type=float,
        default=None,
        help="Optional sun light intensity multiplier (default from preset/config).",
    )
    parser.add_argument(
        "--fill-intensity",
        type=float,
        default=None,
        help="Optional fill/street light intensity multiplier (default from preset/config).",
    )
    parser.add_argument(
        "--shadow-strength",
        type=float,
        default=None,
        help="Optional shadow strength in [0,1]. 0 disables darkening while keeping direct light.",
    )
    parser.add_argument(
        "--disable-shadows",
        action="store_true",
        help="Disable directional shadow mapping.",
    )
    parser.add_argument(
        "--street-light-intensity",
        type=float,
        default=None,
        help="Optional street-light intensity multiplier. Useful for night scenes.",
    )
    parser.add_argument(
        "--street-light-count",
        type=int,
        default=8,
        help="Number of procedural street lights distributed along the road.",
    )
    parser.add_argument(
        "--traffic-flow",
        type=str,
        choices=["into_scene", "toward_camera"],
        default=None,
        help="Traffic movement direction. Default reads from scene config.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    object_dir = SAMPLE_ROOT / "object"

    if args.extract:
        print("[BTL2] Extracting required archives...")
        extract_archives(object_dir=object_dir, selected_archives=REQUIRED_ARCHIVES, overwrite=False)
        print("[BTL2] Archive extraction done.")

    generator = BTL2DatasetGenerator(
        sample_root=SAMPLE_ROOT,
        output_root=args.output,
        width=args.width,
        height=args.height,
        scene_config_path=args.scene_config,
        max_faces=args.max_faces,
        rgb_source=args.rgb_source,
        gaussian_rgb_dir=(args.gaussian_rgb_dir.resolve() if args.gaussian_rgb_dir is not None else None),
        gaussian_blend=float(args.gaussian_blend),
        fps=args.fps,
        scene_mode=args.scene_mode,
        write_video=(not args.no_video),
        time_of_day=args.time_of_day,
        camera_view=args.camera_view,
        sun_azimuth_deg=args.sun_azimuth,
        sun_elevation_deg=args.sun_elevation,
        sun_intensity=args.sun_intensity,
        fill_intensity=args.fill_intensity,
        shadow_strength=args.shadow_strength,
        disable_shadows=args.disable_shadows,
        street_light_intensity=args.street_light_intensity,
        street_light_count=args.street_light_count,
        traffic_flow=args.traffic_flow,
    )
    try:
        summary = generator.generate(frames=args.frames, seed=args.seed)
    finally:
        generator.close()

    print("[BTL2] Generation complete.")
    print(f"  output       : {args.output}")
    print(f"  images       : {summary['images']}")
    print(f"  annotations  : {summary['annotations']}")
    print(f"  classes      : {summary['classes']}")
    if "gaussian_rgb_hits" in summary:
        print(f"  rgb_source   : {summary['rgb_source']}")
        print(f"  gauss_hits   : {summary['gaussian_rgb_hits']}")
        print(f"  gauss_misses : {summary['gaussian_rgb_misses']}")
    if "video_mode" in summary:
        print(f"  scene_mode   : {summary['scene_mode']}")
        if "camera_profile" in summary:
            print(f"  camera_view  : {summary['camera_profile']}")
        if "time_of_day" in summary:
            print(f"  time_of_day  : {summary['time_of_day']}")
        if "traffic_flow" in summary:
            print(f"  traffic_flow : {summary['traffic_flow']}")
        if "street_light_intensity" in summary:
            print(f"  street_light : {summary['street_light_intensity']:.2f} (count={summary.get('street_light_count', 0)})")
        print(f"  video_mode   : {summary['video_mode']}")
        for key in ("rgb_video", "segment_video", "detection_video", "triplet_video"):
            if key in summary:
                print(f"  {key:12}: {summary[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
