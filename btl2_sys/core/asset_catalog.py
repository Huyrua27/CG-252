from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


CLASS_NAMES: List[str] = [
    "car",
    "truck",
    "bike",
    "moto",
    "person",
    "tree",
    "road",
    "building",
]

CLASS_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}


@dataclass
class AssetEntry:
    key: str
    category: str
    obj_relpath: Optional[str] = None
    billboard_texture_relpath: Optional[str] = None
    forced_texture_relpaths: List[str] = field(default_factory=list)
    reference_globs: List[str] = field(default_factory=list)
    max_faces: Optional[int] = 1_000_000
    keep_largest_component: bool = False
    remove_outliers: bool = False
    scale_mode: str = "length"  # length | height | fixed
    target_size: float = 1.0
    fixed_scale: Optional[List[float]] = None
    base_rotation_deg: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    base_offset: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    notes: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


def default_assets() -> Dict[str, AssetEntry]:
    """
    Curated set of assets for road scene generation.
    Paths are relative to `Sample/`.
    """
    return {
        "car_audi_r8": AssetEntry(
            key="car_audi_r8",
            category="car",
            obj_relpath=(
                "object/extracted/xihu59td0bnk-AudiR8Spyder_2017/"
                "3D Models/Audi_R8_2017.obj"
            ),
            reference_globs=[
                "object/extracted/xihu59td0bnk-AudiR8Spyder_2017/Renders/*.png"
            ],
            max_faces=700_000,
            scale_mode="length",
            target_size=4.6,
            notes="High-detail car mesh with render references.",
        ),
        "car_mercedes": AssetEntry(
            key="car_mercedes",
            category="car",
            obj_relpath=(
                "object/extracted/32-mercedes-benz-gls-580-2020/"
                "uploads_files_2787791_Mercedes+Benz+GLS+580.obj"
            ),
            max_faces=250_000,
            scale_mode="length",
            target_size=5.0,
            base_offset=[0.0, 0.06, 0.0],
        ),
        "truck_main": AssetEntry(
            key="truck_main",
            category="truck",
            obj_relpath="object/extracted/64-truck/untitled_trianglefaced.obj",
            reference_globs=["object/extracted/64-truck/Truck.png"],
            max_faces=100_000,
            scale_mode="length",
            target_size=8.2,
        ),
        "bike_mountain": AssetEntry(
            key="bike_mountain",
            category="bike",
            obj_relpath=(
                "object/extracted/54-mountain_bike/Mountain_Bike/OBJ/Mountain_Bike.obj"
            ),
            max_faces=100_000,
            remove_outliers=True,
            scale_mode="length",
            target_size=1.85,
        ),
        "moto_simple": AssetEntry(
            key="moto_simple",
            category="moto",
            obj_relpath="object/extracted/22-moto_simple/moto_simple_1.obj",
            max_faces=50_000,
            remove_outliers=True,
            scale_mode="length",
            target_size=2.1,
        ),
        "tree_tyro": AssetEntry(
            key="tree_tyro",
            category="tree",
            obj_relpath=(
                "object/extracted/1f9jtr180dxk-Tree1ByTyroSmith/Tree1/Tree1.obj"
            ),
            reference_globs=[
                "object/extracted/1f9jtr180dxk-Tree1ByTyroSmith/Tree1/render1hdcool.png"
            ],
            max_faces=450_000,
            remove_outliers=True,
            scale_mode="height",
            target_size=6.0,
        ),
        "road_street": AssetEntry(
            key="road_street",
            category="road",
            obj_relpath="object/extracted/80-street/rua para blender/untitled.obj",
            reference_globs=[
                "object/extracted/80-street/rua para blender/render*.png"
            ],
            max_faces=50_000,
            forced_texture_relpaths=[
                "object/extracted/80-street/rua para blender/rua com faixada.jpg",
                "object/extracted/80-street/rua para blender/render4.png",
                "object/extracted/80-street/rua para blender/render2.png",
                "object/extracted/80-street/rua para blender/render.png",
            ],
            scale_mode="fixed",
            target_size=1.0,
            fixed_scale=[4.2, 1.0, 2.2],
            base_rotation_deg=[0.0, 90.0, 0.0],
        ),
        "building_city": AssetEntry(
            key="building_city",
            category="building",
            obj_relpath="object/extracted/br0e6h1jamf4-building2/building2/building.obj",
            reference_globs=[
                "object/extracted/br0e6h1jamf4-building2/building2/building*.png"
            ],
            max_faces=100_000,
            remove_outliers=True,
            scale_mode="height",
            target_size=18.0,
        ),
        "person_billboard": AssetEntry(
            key="person_billboard",
            category="person",
            billboard_texture_relpath=(
                "object/extracted/55-rp_nathan_animated_003_walking_fbx/"
                "rp_nathan_animated_003_walking_A.jpg"
            ),
            reference_globs=[
                "object/extracted/55-rp_nathan_animated_003_walking_fbx/"
                "rp_nathan_animated_003_walking_A.jpg"
            ],
            max_faces=2,
            scale_mode="height",
            target_size=1.75,
            notes="Pedestrian billboard because provided person assets are FBX-only.",
        ),
    }


def resolve_catalog(
    sample_root: Path,
    override_max_faces: Optional[int] = None,
) -> Dict[str, AssetEntry]:
    catalog = default_assets()
    sample_root = sample_root.resolve()
    for entry in catalog.values():
        if override_max_faces is not None and entry.obj_relpath is not None:
            if override_max_faces <= 0:
                # Unlimited quality mode: keep every face from OBJ.
                entry.max_faces = None
            elif entry.max_faces is None:
                entry.max_faces = int(override_max_faces)
            else:
                entry.max_faces = min(int(entry.max_faces), int(override_max_faces))
        if entry.obj_relpath is not None:
            obj_path = sample_root / entry.obj_relpath
            if not obj_path.exists():
                raise FileNotFoundError(f"Missing OBJ asset: {obj_path}")
        if entry.billboard_texture_relpath is not None:
            tex_path = sample_root / entry.billboard_texture_relpath
            if not tex_path.exists():
                raise FileNotFoundError(f"Missing billboard texture: {tex_path}")
        for rel in entry.forced_texture_relpaths:
            tex_path = sample_root / rel
            if not tex_path.exists():
                raise FileNotFoundError(f"Missing forced texture: {tex_path}")
    return catalog


def collect_reference_images(sample_root: Path, catalog: Dict[str, AssetEntry]) -> List[Path]:
    refs: List[Path] = []
    sample_root = sample_root.resolve()
    for entry in catalog.values():
        for pattern in entry.reference_globs:
            refs.extend(sorted(sample_root.glob(pattern)))
    # Keep deterministic unique order.
    dedup: List[Path] = []
    seen = set()
    for p in refs:
        key = str(p.resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(p)
    return dedup


def export_catalog_json(
    sample_root: Path,
    catalog: Dict[str, AssetEntry],
    output_path: Path,
) -> None:
    import json

    sample_root = sample_root.resolve()
    payload = {
        "class_names": CLASS_NAMES,
        "assets": [],
    }
    for entry in catalog.values():
        d = entry.to_dict()
        if d["obj_relpath"] is not None:
            d["obj_path"] = str((sample_root / d["obj_relpath"]).resolve())
        if d["billboard_texture_relpath"] is not None:
            d["billboard_texture_path"] = str(
                (sample_root / d["billboard_texture_relpath"]).resolve()
            )
        payload["assets"].append(d)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
