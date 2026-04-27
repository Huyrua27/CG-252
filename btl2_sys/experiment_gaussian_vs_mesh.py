from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


THIS_FILE = Path(__file__).resolve()
SAMPLE_ROOT = THIS_FILE.parent.parent.resolve()
if str(SAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(SAMPLE_ROOT))

from btl2_sys.core.asset_catalog import CLASS_NAMES, AssetEntry, default_assets


@dataclass
class ExperimentPaths:
    root: Path
    baseline_out: Path
    hybrid_out: Path
    gaussian_workspace: Path
    gaussian_images: Path
    gaussian_model: Path
    gaussian_frames: Path
    report_json: Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run baseline mesh-only vs Gaussian-hybrid experiment: "
            "pick object with most references, train one Gaussian model, render 100-300 frames, compare mAP."
        )
    )
    parser.add_argument("--frames", type=int, default=150, help="Number of frames to generate (recommended 100-300).")
    parser.add_argument("--width", type=int, default=640, help="Render width.")
    parser.add_argument("--height", type=int, default=360, help="Render height.")
    parser.add_argument("--seed", type=int, default=252, help="Random seed for both datasets.")
    parser.add_argument(
        "--asset-key",
        type=str,
        default=None,
        help="Asset key to use for Gaussian training. If omitted, auto-selects asset with most reference images.",
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=SAMPLE_ROOT / "btl2_sys" / "experiments" / "gaussian_vs_mesh",
        help="Root directory for experiment outputs.",
    )
    parser.add_argument(
        "--gaussian-train-cmd",
        type=str,
        default=None,
        help=(
            "Optional shell command to train Gaussian model. Supported placeholders: "
            "{images_dir}, {workspace_dir}, {model_dir}, {asset_key}, {sample_root}."
        ),
    )
    parser.add_argument(
        "--gaussian-render-cmd",
        type=str,
        default=None,
        help=(
            "Optional shell command to render Gaussian RGB frames. Supported placeholders: "
            "{model_dir}, {frames_dir}, {camera_meta_dir}, {width}, {height}, {frames}, {sample_root}."
        ),
    )
    parser.add_argument(
        "--gaussian-rgb-dir",
        type=Path,
        default=None,
        help="Optional existing Gaussian RGB directory (000001.png...). If missing, --gaussian-render-cmd should generate it.",
    )
    parser.add_argument(
        "--gaussian-blend",
        type=float,
        default=1.0,
        help="Blend factor for Gaussian RGB in hybrid mode [0,1].",
    )
    parser.add_argument(
        "--compare-map",
        action="store_true",
        help="Train YOLO and compare mAP for baseline vs hybrid datasets.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Training epochs for mAP comparison (when --compare-map is enabled).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO image size for mAP comparison.",
    )
    parser.add_argument(
        "--val-data-yaml",
        type=Path,
        default=None,
        help="Optional external validation dataset yaml. If omitted, uses each synthetic dataset itself as validation.",
    )
    return parser.parse_args()


def _count_reference_images(sample_root: Path, entry: AssetEntry) -> int:
    total = 0
    for pattern in entry.reference_globs:
        total += len(list(sample_root.glob(pattern)))
    return total


def _select_asset(sample_root: Path, requested_key: Optional[str]) -> Tuple[str, AssetEntry, int]:
    assets = default_assets()
    if requested_key:
        if requested_key not in assets:
            raise KeyError(f"Unknown asset key: {requested_key}")
        entry = assets[requested_key]
        return requested_key, entry, _count_reference_images(sample_root, entry)

    ranked: List[Tuple[int, str, AssetEntry]] = []
    for key, entry in assets.items():
        count = _count_reference_images(sample_root, entry)
        ranked.append((count, key, entry))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)

    if not ranked or ranked[0][0] <= 0:
        raise RuntimeError("No reference images found in asset_catalog reference_globs.")

    best_count, best_key, best_entry = ranked[0]
    return best_key, best_entry, best_count


def _run_shell(command: str, cwd: Path) -> None:
    print(f"[EXP] $ {command}")
    subprocess.run(command, cwd=str(cwd), shell=True, check=True)


def _copy_asset_references(sample_root: Path, entry: AssetEntry, dst_dir: Path) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for pattern in entry.reference_globs:
        for src in sorted(sample_root.glob(pattern)):
            if not src.is_file():
                continue
            dst = dst_dir / src.name
            if dst.exists():
                continue
            shutil.copy2(src, dst)
            copied += 1
    return copied


def _build_paths(experiment_root: Path) -> ExperimentPaths:
    root = experiment_root.resolve()
    return ExperimentPaths(
        root=root,
        baseline_out=root / "dataset_mesh_only",
        hybrid_out=root / "dataset_gaussian_hybrid",
        gaussian_workspace=root / "gaussian_workspace",
        gaussian_images=root / "gaussian_workspace" / "images",
        gaussian_model=root / "gaussian_workspace" / "model",
        gaussian_frames=root / "gaussian_workspace" / "renders_rgb",
        report_json=root / "report.json",
    )


def _run_dataset_generation(
    sample_root: Path,
    out_dir: Path,
    frames: int,
    width: int,
    height: int,
    seed: int,
    rgb_source: str,
    gaussian_rgb_dir: Optional[Path],
    gaussian_blend: float,
) -> None:
    cmd = [
        sys.executable,
        str(sample_root / "btl2_sys" / "run.py"),
        "--frames",
        str(frames),
        "--width",
        str(width),
        "--height",
        str(height),
        "--seed",
        str(seed),
        "--output",
        str(out_dir),
        "--rgb-source",
        rgb_source,
        "--gaussian-blend",
        str(gaussian_blend),
    ]
    if gaussian_rgb_dir is not None:
        cmd.extend(["--gaussian-rgb-dir", str(gaussian_rgb_dir)])

    print("[EXP] Running dataset generation:")
    print("       " + " ".join(cmd))
    subprocess.run(cmd, cwd=str(sample_root), check=True)


def _write_train_yaml(dataset_root: Path, out_yaml: Path) -> None:
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))
    payload = (
        f"path: {dataset_root.as_posix()}\n"
        f"train: images/rgb\n"
        f"val: images/rgb\n"
        f"names:\n{names}\n"
    )
    out_yaml.write_text(payload, encoding="utf-8")


def _train_and_eval_map(
    dataset_root: Path,
    run_root: Path,
    run_name: str,
    epochs: int,
    imgsz: int,
    val_data_yaml: Optional[Path],
) -> Dict[str, float]:
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "ultralytics is required for --compare-map. Install it with: pip install ultralytics"
        ) from exc

    run_root.mkdir(parents=True, exist_ok=True)
    train_yaml = run_root / f"{run_name}_train.yaml"
    _write_train_yaml(dataset_root, train_yaml)

    model = YOLO("yolov8n.pt")
    model.train(
        data=str(train_yaml),
        epochs=int(epochs),
        imgsz=int(imgsz),
        project=str(run_root),
        name=f"{run_name}_train",
        exist_ok=True,
        verbose=False,
    )

    eval_data = str(val_data_yaml) if val_data_yaml is not None else str(train_yaml)
    metrics = model.val(data=eval_data, imgsz=int(imgsz), verbose=False)

    map50 = float(getattr(metrics.box, "map50", 0.0))
    map50_95 = float(getattr(metrics.box, "map", 0.0))
    return {"map50": map50, "map50_95": map50_95}


def _resolve_gaussian_rgb_dir(args: argparse.Namespace, paths: ExperimentPaths) -> Path:
    if args.gaussian_rgb_dir is not None:
        return args.gaussian_rgb_dir.resolve()
    return paths.gaussian_frames


def main() -> int:
    args = _parse_args()
    if args.frames < 100 or args.frames > 300:
        print("[EXP] Warning: recommended range is 100-300 frames.")

    paths = _build_paths(args.experiment_root)
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.gaussian_workspace.mkdir(parents=True, exist_ok=True)

    asset_key, asset_entry, ref_count = _select_asset(SAMPLE_ROOT, args.asset_key)
    print(f"[EXP] Selected asset: {asset_key} (class={asset_entry.category}, refs={ref_count})")

    copied = _copy_asset_references(SAMPLE_ROOT, asset_entry, paths.gaussian_images)
    print(f"[EXP] Prepared Gaussian training references in {paths.gaussian_images} (copied={copied}).")

    if args.gaussian_train_cmd:
        train_cmd = args.gaussian_train_cmd.format(
            images_dir=str(paths.gaussian_images),
            workspace_dir=str(paths.gaussian_workspace),
            model_dir=str(paths.gaussian_model),
            asset_key=asset_key,
            sample_root=str(SAMPLE_ROOT),
        )
        _run_shell(train_cmd, cwd=SAMPLE_ROOT)
    else:
        print("[EXP] gaussian-train-cmd not provided: skipping model training command.")

    gaussian_rgb_dir = _resolve_gaussian_rgb_dir(args, paths)
    if args.gaussian_render_cmd:
        render_cmd = args.gaussian_render_cmd.format(
            model_dir=str(paths.gaussian_model),
            frames_dir=str(gaussian_rgb_dir),
            camera_meta_dir=str(paths.baseline_out / "metadata" / "scenes"),
            width=str(args.width),
            height=str(args.height),
            frames=str(args.frames),
            sample_root=str(SAMPLE_ROOT),
        )
        _run_shell(render_cmd, cwd=SAMPLE_ROOT)

    # Baseline mesh-only generation.
    _run_dataset_generation(
        sample_root=SAMPLE_ROOT,
        out_dir=paths.baseline_out,
        frames=args.frames,
        width=args.width,
        height=args.height,
        seed=args.seed,
        rgb_source="mesh",
        gaussian_rgb_dir=None,
        gaussian_blend=1.0,
    )

    # Hybrid generation: Gaussian RGB + mesh GT.
    _run_dataset_generation(
        sample_root=SAMPLE_ROOT,
        out_dir=paths.hybrid_out,
        frames=args.frames,
        width=args.width,
        height=args.height,
        seed=args.seed,
        rgb_source="gaussian_hybrid",
        gaussian_rgb_dir=gaussian_rgb_dir,
        gaussian_blend=float(args.gaussian_blend),
    )

    report: Dict[str, object] = {
        "selected_asset": {
            "asset_key": asset_key,
            "category": asset_entry.category,
            "reference_image_count": ref_count,
        },
        "datasets": {
            "mesh_only": str(paths.baseline_out),
            "gaussian_hybrid": str(paths.hybrid_out),
            "gaussian_rgb_dir": str(gaussian_rgb_dir),
        },
        "map": None,
    }

    if args.compare_map:
        print("[EXP] Training and evaluating mAP (this can take a while)...")
        run_root = paths.root / "map_runs"
        baseline_map = _train_and_eval_map(
            dataset_root=paths.baseline_out,
            run_root=run_root,
            run_name="mesh_only",
            epochs=args.epochs,
            imgsz=args.imgsz,
            val_data_yaml=(args.val_data_yaml.resolve() if args.val_data_yaml is not None else None),
        )
        hybrid_map = _train_and_eval_map(
            dataset_root=paths.hybrid_out,
            run_root=run_root,
            run_name="gaussian_hybrid",
            epochs=args.epochs,
            imgsz=args.imgsz,
            val_data_yaml=(args.val_data_yaml.resolve() if args.val_data_yaml is not None else None),
        )

        delta_map50 = hybrid_map["map50"] - baseline_map["map50"]
        delta_map50_95 = hybrid_map["map50_95"] - baseline_map["map50_95"]

        report["map"] = {
            "mesh_only": baseline_map,
            "gaussian_hybrid": hybrid_map,
            "delta": {
                "map50": delta_map50,
                "map50_95": delta_map50_95,
            },
        }
        print(
            "[EXP] mAP delta (hybrid - mesh): "
            f"map50={delta_map50:.4f}, map50_95={delta_map50_95:.4f}"
        )

    paths.report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[EXP] Done. Report: {paths.report_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
