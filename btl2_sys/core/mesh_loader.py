from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class MaterialData:
    name: str
    kd: np.ndarray
    ks: np.ndarray
    ka: np.ndarray
    shininess: float
    texture_path: Optional[Path] = None


@dataclass
class MeshData:
    name: str
    vertices: np.ndarray  # (N, 3) float32
    normals: np.ndarray  # (N, 3) float32
    uvs: np.ndarray  # (N, 2) float32
    triangles: np.ndarray  # (T, 3) uint32
    tri_material_ids: np.ndarray  # (T,) int32
    materials: List[MaterialData]
    bbox_min: np.ndarray  # (3,)
    bbox_max: np.ndarray  # (3,)

    @property
    def extents(self) -> np.ndarray:
        return self.bbox_max - self.bbox_min


def _normalize_rows(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms = np.where(norms < eps, 1.0, norms)
    return values / norms


def _parse_rgb(parts: Sequence[str], default: Tuple[float, float, float]) -> np.ndarray:
    if len(parts) < 4:
        return np.asarray(default, dtype=np.float32)
    try:
        return np.asarray([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float32)
    except ValueError:
        return np.asarray(default, dtype=np.float32)


def _collect_image_files(root: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg"}
    resolved_root = root.resolve()

    def _should_scan_parent_dir(obj_dir: Path) -> bool:
        key = obj_dir.name.lower().replace("_", " ").replace("-", " ").strip()
        return key in {"obj", "objects", "3d models", "3d model", "models", "model", "mesh", "meshes"}

    roots = [resolved_root]
    parent = resolved_root.parent
    if parent != resolved_root and _should_scan_parent_dir(resolved_root):
        roots.append(parent)

    out: List[Path] = []
    seen = set()
    for scan_root in roots:
        if not scan_root.exists():
            continue
        for p in scan_root.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in exts:
                continue
            r = p.resolve()
            key = str(r).lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(r)
    out.sort(key=lambda p: str(p).lower())
    return out


def _is_default_gray_kd(kd: np.ndarray) -> bool:
    base = np.asarray([0.64, 0.64, 0.64], dtype=np.float32)
    return bool(np.max(np.abs(np.asarray(kd, dtype=np.float32) - base)) <= 0.035)


def _material_color_hint(material_name: str) -> Optional[np.ndarray]:
    lower = material_name.lower()
    hex_match = re.search(r"(?<![0-9a-fA-F])([0-9a-fA-F]{6})(?![0-9a-fA-F])", material_name)
    if hex_match:
        h = hex_match.group(1)
        rgb = [int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4)]
        return np.asarray(rgb, dtype=np.float32)

    ordered = [
        (("polar", "white"), [0.92, 0.92, 0.92]),
        (("bodycolour",), [0.18, 0.42, 0.76]),
        (("bodycolor",), [0.18, 0.42, 0.76]),
        (("bodymaterials",), [0.18, 0.42, 0.76]),
        (("color_",), [0.20, 0.32, 0.58]),
        (("leaves",), [0.22, 0.52, 0.21]),
        (("leaf",), [0.22, 0.52, 0.21]),
        (("bark",), [0.37, 0.26, 0.17]),
        (("wood",), [0.42, 0.30, 0.20]),
        (("tire",), [0.06, 0.06, 0.06]),
        (("tyre",), [0.06, 0.06, 0.06]),
        (("wheel",), [0.12, 0.12, 0.12]),
        (("glass",), [0.36, 0.39, 0.43]),
        (("chrome",), [0.75, 0.75, 0.75]),
        (("silver",), [0.72, 0.72, 0.72]),
        (("interior",), [0.16, 0.16, 0.17]),
        (("under",), [0.14, 0.14, 0.15]),
        (("black",), [0.08, 0.08, 0.09]),
        (("white",), [0.88, 0.88, 0.89]),
        (("grey",), [0.44, 0.44, 0.45]),
        (("gray",), [0.44, 0.44, 0.45]),
        (("light",), [0.95, 0.87, 0.60]),
        (("signal",), [0.95, 0.66, 0.24]),
        (("red",), [0.75, 0.16, 0.14]),
        (("blue",), [0.16, 0.35, 0.72]),
        (("green",), [0.17, 0.58, 0.24]),
        (("yellow",), [0.78, 0.68, 0.16]),
        (("orange",), [0.78, 0.43, 0.12]),
        (("brown",), [0.40, 0.28, 0.18]),
    ]
    for tokens, color in ordered:
        if all(token in lower for token in tokens):
            return np.asarray(color, dtype=np.float32)
    return None


def _finalize_material_appearance(material: MaterialData) -> None:
    if material.texture_path is not None and _is_default_gray_kd(material.kd):
        material.kd = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
    elif material.texture_path is None and _is_default_gray_kd(material.kd):
        hint = _material_color_hint(material.name)
        if hint is not None:
            material.kd = hint
    ka_floor = np.clip(np.asarray(material.kd, dtype=np.float32) * 0.12, 0.03, 0.22)
    material.ka = np.maximum(np.asarray(material.ka, dtype=np.float32), ka_floor)


def _texture_candidate_score(path: Path, material_name: str) -> int:
    name = path.name.lower()
    full = str(path).lower()
    mat_key = "".join(ch for ch in material_name.lower() if ch.isalnum())
    stem_key = "".join(ch for ch in path.stem.lower() if ch.isalnum())

    score = 0
    if any(k in full for k in ("/texture", "\\texture", "/textures", "\\textures")):
        score += 8
    if any(k in name for k in ("diff", "albedo", "basecolor", "color", "col", "kd")):
        score += 6
    if any(k in name for k in ("render", "preview", "screenshot", "beauty")):
        score -= 18
    if any(k in name for k in ("normal", "rough", "metal", "spec", "gloss", "bump", "height", "ao")):
        score -= 5
    if mat_key and (mat_key in stem_key or stem_key in mat_key):
        score += 4
    return score


def _resolve_texture_path(
    raw_texture_path: Optional[str],
    material_name: str,
    obj_dir: Path,
    image_files: List[Path],
) -> Optional[Path]:
    def _should_scan_parent_dir(obj_dir_path: Path) -> bool:
        key = obj_dir_path.name.lower().replace("_", " ").replace("-", " ").strip()
        return key in {"obj", "objects", "3d models", "3d model", "models", "model", "mesh", "meshes"}

    use_parent_lookup = _should_scan_parent_dir(obj_dir.resolve())
    candidates: List[Path] = []
    if raw_texture_path:
        cleaned = raw_texture_path.strip().strip('"').strip("'").replace("\\", "/")
        option_like = cleaned.startswith("-")
        fallback_name = None
        if option_like:
            tokens = cleaned.split()
            fallback_name = tokens[-1] if tokens else None

        if cleaned:
            candidates.append((obj_dir / cleaned).resolve())
        tex_name = Path(cleaned).name if cleaned else ""
        if tex_name:
            candidates.append((obj_dir / tex_name).resolve())
            if use_parent_lookup:
                candidates.append((obj_dir.parent / tex_name).resolve())
            candidates.extend([p for p in image_files if p.name.lower() == tex_name.lower()])
        if fallback_name:
            candidates.append((obj_dir / fallback_name).resolve())
            if use_parent_lookup:
                candidates.append((obj_dir.parent / fallback_name).resolve())
            candidates.extend([p for p in image_files if p.name.lower() == fallback_name.lower()])

    mat_key = "".join(ch for ch in material_name.lower() if ch.isalnum())
    if mat_key:
        for img in image_files:
            stem_key = "".join(ch for ch in img.stem.lower() if ch.isalnum())
            if mat_key in stem_key or stem_key in mat_key:
                candidates.append(img)

    scored: List[Tuple[int, Path]] = []
    seen = set()
    for cand in candidates:
        if not cand.exists():
            continue
        resolved = cand.resolve()
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        scored.append((_texture_candidate_score(resolved, material_name), resolved))

    if not scored:
        return None
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_path = scored[0]
    if best_score < -6:
        return None
    return best_path


def load_mtl(mtl_path: Path, obj_dir: Path) -> Dict[str, MaterialData]:
    if not mtl_path.exists():
        return {}
    image_files = _collect_image_files(obj_dir)
    materials: Dict[str, MaterialData] = {}
    current: Optional[MaterialData] = None
    raw_texture: Optional[str] = None

    for line in mtl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        head = parts[0].lower()
        if head == "newmtl":
            if current is not None:
                current.texture_path = _resolve_texture_path(raw_texture, current.name, obj_dir, image_files)
                _finalize_material_appearance(current)
                materials[current.name] = current
            mat_name = parts[1] if len(parts) > 1 else f"material_{len(materials)}"
            current = MaterialData(
                name=mat_name,
                kd=np.asarray([0.64, 0.64, 0.64], dtype=np.float32),
                ks=np.asarray([0.45, 0.45, 0.45], dtype=np.float32),
                ka=np.asarray([0.08, 0.08, 0.08], dtype=np.float32),
                shininess=48.0,
                texture_path=None,
            )
            raw_texture = None
            continue
        if current is None:
            continue
        if head == "kd":
            current.kd = _parse_rgb(parts, (0.64, 0.64, 0.64))
        elif head == "ks":
            current.ks = _parse_rgb(parts, (0.45, 0.45, 0.45))
        elif head == "ka":
            current.ka = _parse_rgb(parts, (0.08, 0.08, 0.08))
        elif head == "ns":
            try:
                current.shininess = max(2.0, min(256.0, float(parts[1])))
            except (IndexError, ValueError):
                current.shininess = 48.0
        elif head in {"map_kd"}:
            raw_texture = stripped.split(" ", 1)[1] if " " in stripped else None

    if current is not None:
        current.texture_path = _resolve_texture_path(raw_texture, current.name, obj_dir, image_files)
        _finalize_material_appearance(current)
        materials[current.name] = current
    return materials


def _default_material(name: str) -> MaterialData:
    hint = _material_color_hint(name)
    if hint is not None:
        kd = np.clip(hint, 0.03, 0.95)
    else:
        seed = abs(hash(name)) % 10000
        base = np.asarray(
            [
                0.35 + (seed % 31) / 100.0,
                0.35 + ((seed // 7) % 31) / 100.0,
                0.35 + ((seed // 13) % 31) / 100.0,
            ],
            dtype=np.float32,
        )
        kd = np.clip(base, 0.2, 0.9)
    return MaterialData(
        name=name,
        kd=kd,
        ks=np.asarray([0.25, 0.25, 0.25], dtype=np.float32),
        ka=np.asarray([0.06, 0.06, 0.06], dtype=np.float32),
        shininess=36.0,
        texture_path=None,
    )


def _parse_face_index(token: str, n_v: int, n_vt: int, n_vn: int) -> Tuple[int, int, int]:
    fields = token.split("/")
    vi = int(fields[0]) if fields and fields[0] else 0
    ti = int(fields[1]) if len(fields) > 1 and fields[1] else 0
    ni = int(fields[2]) if len(fields) > 2 and fields[2] else 0

    if vi < 0:
        vi = n_v + vi + 1
    if ti < 0:
        ti = n_vt + ti + 1
    if ni < 0:
        ni = n_vn + ni + 1

    return vi - 1, (ti - 1 if ti else -1), (ni - 1 if ni else -1)


def _compact_mesh(mesh: MeshData) -> MeshData:
    if mesh.triangles.size == 0:
        return mesh
    used = np.unique(mesh.triangles.reshape(-1))
    remap = np.full(mesh.vertices.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.shape[0], dtype=np.int64)

    triangles = remap[mesh.triangles]
    vertices = mesh.vertices[used]
    normals = mesh.normals[used]
    uvs = mesh.uvs[used]

    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    return MeshData(
        name=mesh.name,
        vertices=vertices.astype(np.float32),
        normals=normals.astype(np.float32),
        uvs=uvs.astype(np.float32),
        triangles=triangles.astype(np.uint32),
        tri_material_ids=mesh.tri_material_ids.astype(np.int32),
        materials=mesh.materials,
        bbox_min=bbox_min.astype(np.float32),
        bbox_max=bbox_max.astype(np.float32),
    )


def _largest_component_triangle_mask(triangles: np.ndarray) -> np.ndarray:
    if triangles.size == 0:
        return np.zeros((0,), dtype=bool)
    n_vertices = int(triangles.max()) + 1
    parent = np.arange(n_vertices, dtype=np.int64)
    rank = np.zeros(n_vertices, dtype=np.int8)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    for tri in triangles:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        union(a, b)
        union(b, c)

    tri_roots = np.asarray([find(int(tri[0])) for tri in triangles], dtype=np.int64)
    unique_roots, counts = np.unique(tri_roots, return_counts=True)
    largest_root = unique_roots[np.argmax(counts)]
    return tri_roots == largest_root


def _outlier_triangle_mask(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    if triangles.shape[0] < 200:
        return np.ones((triangles.shape[0],), dtype=bool)
    centers = vertices[triangles].mean(axis=1)
    lo = np.percentile(centers, 1.0, axis=0)
    hi = np.percentile(centers, 99.0, axis=0)
    margin = np.maximum((hi - lo) * 0.35, 1e-4)
    keep = np.all((centers >= (lo - margin)) & (centers <= (hi + margin)), axis=1)
    if keep.mean() < 0.45:
        return np.ones((triangles.shape[0],), dtype=bool)
    return keep


def _select_triangle_subset(vertices: np.ndarray, triangles: np.ndarray, max_faces: int) -> np.ndarray:
    if triangles.shape[0] <= max_faces:
        return np.arange(triangles.shape[0], dtype=np.int64)
    p0 = vertices[triangles[:, 0]]
    p1 = vertices[triangles[:, 1]]
    p2 = vertices[triangles[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    order = np.argsort(areas, kind="stable")
    keep = np.sort(order[-max_faces:])
    return keep.astype(np.int64)


def _is_helper_material_name(material_name: str) -> bool:
    key = re.sub(r"[^a-z0-9]+", "", material_name.lower())
    helper_tokens = ("ground", "shadow", "backdrop", "background", "floor", "helper", "dummy", "matte")
    return any(token in key for token in helper_tokens)


def _helper_material_triangle_mask(tri_mat_names: List[str], max_faces_per_helper: int = 32) -> np.ndarray:
    if not tri_mat_names:
        return np.ones((0,), dtype=bool)
    counts: Dict[str, int] = {}
    for name in tri_mat_names:
        counts[name] = counts.get(name, 0) + 1

    keep = np.ones((len(tri_mat_names),), dtype=bool)
    for i, name in enumerate(tri_mat_names):
        if _is_helper_material_name(name) and counts.get(name, 0) <= int(max_faces_per_helper):
            keep[i] = False
    return keep


def _triangle_component_roots_by_position(
    vertices: np.ndarray,
    triangles: np.ndarray,
    quant_eps: float = 1e-6,
) -> np.ndarray:
    if triangles.size == 0:
        return np.zeros((0,), dtype=np.int64)
    quantized = np.round(vertices / float(max(quant_eps, 1e-9))).astype(np.int64)
    _, inverse = np.unique(quantized, axis=0, return_inverse=True)
    tri_nodes = inverse[triangles]
    n_nodes = int(inverse.max()) + 1

    parent = np.arange(n_nodes, dtype=np.int64)
    rank = np.zeros(n_nodes, dtype=np.int8)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    for tri in tri_nodes:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        union(a, b)
        union(b, c)

    return np.asarray([find(int(tri[0])) for tri in tri_nodes], dtype=np.int64)


def remove_planar_artifact_components(
    mesh: MeshData,
    max_component_faces: int = 5200,
    max_thickness_ratio: float = 0.06,
    min_major_extent: float = 0.16,
) -> MeshData:
    if mesh.triangles.shape[0] <= 0:
        return mesh
    roots = _triangle_component_roots_by_position(mesh.vertices, mesh.triangles, quant_eps=1e-6)
    if roots.shape[0] != mesh.triangles.shape[0]:
        return mesh

    keep = np.ones((mesh.triangles.shape[0],), dtype=bool)
    unique_roots, counts = np.unique(roots, return_counts=True)
    for root, count in zip(unique_roots.tolist(), counts.tolist()):
        if int(count) > int(max_component_faces):
            continue
        tri_idx = np.where(roots == int(root))[0]
        if tri_idx.size == 0:
            continue
        comp_vertices = mesh.vertices[mesh.triangles[tri_idx].reshape(-1)]
        ext = comp_vertices.max(axis=0) - comp_vertices.min(axis=0)
        ext_sorted = np.sort(ext.astype(np.float64))
        major_a = float(ext_sorted[1])
        major_b = float(ext_sorted[2])
        if major_a < float(min_major_extent) or major_b < float(min_major_extent):
            continue
        thickness = float(ext_sorted[0])
        ratio = thickness / max(major_b, 1e-8)
        if ratio <= float(max_thickness_ratio):
            keep[tri_idx] = False

    if not np.any(~keep) or not np.any(keep):
        return mesh
    filtered = MeshData(
        name=mesh.name,
        vertices=mesh.vertices.copy(),
        normals=mesh.normals.copy(),
        uvs=mesh.uvs.copy(),
        triangles=mesh.triangles[keep].copy(),
        tri_material_ids=mesh.tri_material_ids[keep].copy(),
        materials=mesh.materials,
        bbox_min=mesh.bbox_min.copy(),
        bbox_max=mesh.bbox_max.copy(),
    )
    return _compact_mesh(filtered)


def remove_tiny_high_material_components(
    mesh: MeshData,
    material_name_tokens: Sequence[str],
    max_component_faces: int = 12,
    min_height_quantile: float = 0.84,
) -> MeshData:
    """
    Remove tiny disconnected components that belong to specific helper materials
    and float unusually high above the main mesh body.

    This is useful for assets that embed studio helper planes (e.g. light cards)
    inside the OBJ and accidentally show up in generated scenes.
    """
    if mesh.triangles.shape[0] <= 0:
        return mesh
    if mesh.tri_material_ids is None or mesh.tri_material_ids.shape[0] != mesh.triangles.shape[0]:
        return mesh

    tokens = [re.sub(r"[^a-z0-9]+", "", t.lower()) for t in material_name_tokens if str(t).strip()]
    if not tokens:
        return mesh

    roots = _triangle_component_roots_by_position(mesh.vertices, mesh.triangles, quant_eps=1e-6)
    if roots.shape[0] != mesh.triangles.shape[0]:
        return mesh

    global_y = mesh.vertices[:, 1].astype(np.float64)
    y_threshold = float(np.quantile(global_y, float(np.clip(min_height_quantile, 0.0, 1.0))))

    keep = np.ones((mesh.triangles.shape[0],), dtype=bool)
    unique_roots = np.unique(roots)

    for root in unique_roots.tolist():
        tri_idx = np.where(roots == int(root))[0]
        tri_count = int(tri_idx.size)
        if tri_count <= 0 or tri_count > int(max_component_faces):
            continue

        comp_vertices = mesh.vertices[mesh.triangles[tri_idx].reshape(-1)]
        comp_center_y = float(comp_vertices[:, 1].mean())
        if comp_center_y < y_threshold:
            continue

        comp_mat_ids = mesh.tri_material_ids[tri_idx].astype(np.int64)
        if comp_mat_ids.size <= 0:
            continue
        uniq_mids, mid_counts = np.unique(comp_mat_ids, return_counts=True)
        dominant_mid = int(uniq_mids[np.argmax(mid_counts)])
        if dominant_mid < 0 or dominant_mid >= len(mesh.materials):
            continue

        mat_name = re.sub(r"[^a-z0-9]+", "", mesh.materials[dominant_mid].name.lower())
        if not any(token in mat_name for token in tokens):
            continue

        keep[tri_idx] = False

    if not np.any(~keep) or not np.any(keep):
        return mesh

    filtered = MeshData(
        name=mesh.name,
        vertices=mesh.vertices.copy(),
        normals=mesh.normals.copy(),
        uvs=mesh.uvs.copy(),
        triangles=mesh.triangles[keep].copy(),
        tri_material_ids=mesh.tri_material_ids[keep].copy(),
        materials=mesh.materials,
        bbox_min=mesh.bbox_min.copy(),
        bbox_max=mesh.bbox_max.copy(),
    )
    return _compact_mesh(filtered)


def keep_upward_facing_triangles(mesh: MeshData, min_up_dot: float = 0.85) -> MeshData:
    if mesh.triangles.shape[0] <= 0:
        return mesh
    tri = mesh.triangles
    p0 = mesh.vertices[tri[:, 0]]
    p1 = mesh.vertices[tri[:, 1]]
    p2 = mesh.vertices[tri[:, 2]]
    n = np.cross(p1 - p0, p2 - p0)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln = np.where(ln < 1e-8, 1.0, ln)
    n = n / ln
    keep = n[:, 1] >= float(min_up_dot)
    if not np.any(keep) or np.all(keep):
        return mesh
    filtered = MeshData(
        name=mesh.name,
        vertices=mesh.vertices.copy(),
        normals=mesh.normals.copy(),
        uvs=mesh.uvs.copy(),
        triangles=mesh.triangles[keep].copy(),
        tri_material_ids=mesh.tri_material_ids[keep].copy(),
        materials=mesh.materials,
        bbox_min=mesh.bbox_min.copy(),
        bbox_max=mesh.bbox_max.copy(),
    )
    return _compact_mesh(filtered)


def load_obj(
    path: Path,
    max_faces: Optional[int] = None,
    keep_largest_component: bool = True,
    remove_outliers: bool = True,
) -> MeshData:
    path = path.resolve()
    obj_dir = path.parent

    raw_v: List[Tuple[float, float, float]] = []
    raw_vt: List[Tuple[float, float]] = []
    raw_vn: List[Tuple[float, float, float]] = []
    mtllib_names: List[str] = []
    face_count = 0

    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            head = parts[0].lower()
            if head == "v" and len(parts) >= 4:
                raw_v.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif head == "vt" and len(parts) >= 3:
                raw_vt.append((float(parts[1]), float(parts[2])))
            elif head == "vn" and len(parts) >= 4:
                raw_vn.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif head == "mtllib" and len(parts) >= 2:
                mtllib_names.append(stripped.split(" ", 1)[1].strip())
            elif head == "f" and len(parts) >= 4:
                face_count += max(0, len(parts) - 3)

    if not raw_v or face_count <= 0:
        raise ValueError(f"OBJ has no geometry: {path}")

    materials_lookup: Dict[str, MaterialData] = {}
    for mtl_name in mtllib_names:
        mtl_path = (obj_dir / mtl_name).resolve()
        if not mtl_path.exists():
            mtl_path = (obj_dir / Path(mtl_name).name).resolve()
        materials_lookup.update(load_mtl(mtl_path, obj_dir))
    image_files = _collect_image_files(obj_dir)

    raw_v_arr = np.asarray(raw_v, dtype=np.float32)
    raw_vt_arr = np.asarray(raw_vt, dtype=np.float32) if raw_vt else np.zeros((0, 2), dtype=np.float32)
    raw_vn_arr = np.asarray(raw_vn, dtype=np.float32) if raw_vn else np.zeros((0, 3), dtype=np.float32)

    vertex_cache: Dict[Tuple[int, int, int], int] = {}
    out_v: List[Tuple[float, float, float]] = []
    out_vt: List[Tuple[float, float]] = []
    out_vn: List[Tuple[float, float, float]] = []
    tri_idx: List[Tuple[int, int, int]] = []
    tri_mat_names: List[str] = []

    current_material = "default"
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            head = parts[0].lower()
            if head == "usemtl" and len(parts) >= 2:
                current_material = stripped.split(" ", 1)[1].strip()
                continue
            if head != "f" or len(parts) < 4:
                continue

            fv = [_parse_face_index(tok, len(raw_v), len(raw_vt), len(raw_vn)) for tok in parts[1:]]
            for i in range(1, len(fv) - 1):
                tri: List[int] = []
                for key in (fv[0], fv[i], fv[i + 1]):
                    cached = vertex_cache.get(key)
                    if cached is not None:
                        tri.append(cached)
                        continue
                    vi, ti, ni = key
                    pos = raw_v_arr[vi]
                    uv = (
                        raw_vt_arr[ti]
                        if ti >= 0 and ti < raw_vt_arr.shape[0]
                        else np.asarray([0.0, 0.0], dtype=np.float32)
                    )
                    nor = (
                        raw_vn_arr[ni]
                        if ni >= 0 and ni < raw_vn_arr.shape[0]
                        else np.asarray([0.0, 0.0, 0.0], dtype=np.float32)
                    )

                    idx = len(out_v)
                    vertex_cache[key] = idx
                    out_v.append((float(pos[0]), float(pos[1]), float(pos[2])))
                    out_vt.append((float(uv[0]), float(uv[1])))
                    out_vn.append((float(nor[0]), float(nor[1]), float(nor[2])))
                    tri.append(idx)
                tri_idx.append((tri[0], tri[1], tri[2]))
                tri_mat_names.append(current_material)

    vertices = np.asarray(out_v, dtype=np.float32)
    uvs = np.asarray(out_vt, dtype=np.float32)
    normals = np.asarray(out_vn, dtype=np.float32)
    triangles = np.asarray(tri_idx, dtype=np.uint32)

    missing_normals = np.linalg.norm(normals, axis=1) < 1e-8
    if np.any(missing_normals):
        accum = np.zeros_like(normals, dtype=np.float32)
        for tri in triangles:
            p0, p1, p2 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
            face_n = np.cross(p1 - p0, p2 - p0)
            accum[tri[0]] += face_n
            accum[tri[1]] += face_n
            accum[tri[2]] += face_n
        accum = _normalize_rows(accum)
        normals[missing_normals] = accum[missing_normals]
    normals = _normalize_rows(normals)

    if triangles.shape[0] > 0:
        keep_helper = _helper_material_triangle_mask(tri_mat_names, max_faces_per_helper=32)
        if keep_helper.shape[0] == triangles.shape[0] and np.any(~keep_helper) and np.any(keep_helper):
            triangles = triangles[keep_helper]
            tri_mat_names = [tri_mat_names[i] for i, keep in enumerate(keep_helper.tolist()) if keep]

    if remove_outliers and triangles.shape[0] > 0:
        keep_outlier = _outlier_triangle_mask(vertices, triangles)
        triangles = triangles[keep_outlier]
        tri_mat_names = [tri_mat_names[i] for i, keep in enumerate(keep_outlier.tolist()) if keep]

    if keep_largest_component and triangles.shape[0] > 0:
        keep_component = _largest_component_triangle_mask(triangles)
        triangles = triangles[keep_component]
        tri_mat_names = [tri_mat_names[i] for i, keep in enumerate(keep_component.tolist()) if keep]

    if max_faces is not None and triangles.shape[0] > max_faces:
        keep = _select_triangle_subset(vertices, triangles, max_faces=max_faces)
        triangles = triangles[keep]
        tri_mat_names = [tri_mat_names[i] for i in keep.tolist()]

    material_order: List[str] = []
    mat_to_id: Dict[str, int] = {}
    for name in tri_mat_names:
        if name not in mat_to_id:
            mat_to_id[name] = len(material_order)
            material_order.append(name)
    tri_material_ids = np.asarray([mat_to_id[name] for name in tri_mat_names], dtype=np.int32)

    materials: List[MaterialData] = []
    for name in material_order:
        mat = materials_lookup.get(name, _default_material(name))
        if mat.texture_path is None:
            guessed = _resolve_texture_path(None, name, obj_dir, image_files)
            if guessed is not None:
                mat.texture_path = guessed
        _finalize_material_appearance(mat)
        materials.append(mat)

    base_to_texture: Dict[str, Path] = {}
    for mat in materials:
        if mat.texture_path is None:
            continue
        base_key = mat.name.split(".")[0].strip().lower()
        if base_key and base_key not in base_to_texture:
            base_to_texture[base_key] = mat.texture_path

    for mat in materials:
        if mat.texture_path is not None:
            continue
        base_key = mat.name.split(".")[0].strip().lower()
        inherited = base_to_texture.get(base_key)
        if inherited is None:
            continue
        mat.texture_path = inherited
        _finalize_material_appearance(mat)

    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    mesh = MeshData(
        name=path.stem,
        vertices=vertices,
        normals=normals,
        uvs=uvs,
        triangles=triangles,
        tri_material_ids=tri_material_ids,
        materials=materials,
        bbox_min=bbox_min,
        bbox_max=bbox_max,
    )
    return _compact_mesh(mesh)


def build_billboard_mesh(texture_path: Path, name: str = "person_billboard") -> MeshData:
    vertices = np.asarray(
        [
            [-0.5, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [-0.5, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    normals = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    uvs = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    triangles = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    tri_material_ids = np.asarray([0, 0], dtype=np.int32)
    material = MaterialData(
        name="billboard",
        kd=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
        ks=np.asarray([0.02, 0.02, 0.02], dtype=np.float32),
        ka=np.asarray([0.05, 0.05, 0.05], dtype=np.float32),
        shininess=4.0,
        texture_path=texture_path.resolve(),
    )
    return MeshData(
        name=name,
        vertices=vertices,
        normals=normals,
        uvs=uvs,
        triangles=triangles,
        tri_material_ids=tri_material_ids,
        materials=[material],
        bbox_min=vertices.min(axis=0),
        bbox_max=vertices.max(axis=0),
    )


def build_ground_plane_mesh(
    width: float = 52.0,
    depth: float = 110.0,
    color_rgb: Tuple[float, float, float] = (0.34, 0.26, 0.17),
    name: str = "ground_plane",
) -> MeshData:
    hw = float(width) * 0.5
    hd = float(depth) * 0.5
    vertices = np.asarray(
        [
            [-hw, 0.0, -hd],
            [hw, 0.0, -hd],
            [hw, 0.0, hd],
            [-hw, 0.0, hd],
        ],
        dtype=np.float32,
    )
    normals = np.asarray(
        [
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    uvs = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    triangles = np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    tri_material_ids = np.asarray([0, 0], dtype=np.int32)
    kd = np.asarray(color_rgb, dtype=np.float32)
    material = MaterialData(
        name="ground_material",
        kd=kd,
        ks=np.asarray([0.03, 0.03, 0.03], dtype=np.float32),
        ka=np.asarray([0.12, 0.10, 0.08], dtype=np.float32),
        shininess=6.0,
        texture_path=None,
    )
    return MeshData(
        name=name,
        vertices=vertices,
        normals=normals,
        uvs=uvs,
        triangles=triangles,
        tri_material_ids=tri_material_ids,
        materials=[material],
        bbox_min=vertices.min(axis=0),
        bbox_max=vertices.max(axis=0),
    )
