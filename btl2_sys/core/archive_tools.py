from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path
from typing import Iterable, List, Optional


def _extract_zip(archive_path: Path, output_dir: Path) -> None:
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(output_dir)


def _extract_rar(archive_path: Path, output_dir: Path) -> None:
    # Windows `tar.exe` in this environment can unpack .rar files.
    proc = subprocess.run(
        ["tar", "-xf", str(archive_path), "-C", str(output_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Failed to extract RAR '{archive_path.name}': {proc.stderr.strip()}"
        )


def extract_archives(
    object_dir: Path,
    selected_archives: Optional[Iterable[str]] = None,
    overwrite: bool = False,
) -> List[Path]:
    """
    Extract zip/rar archives from `object_dir` into `object_dir/extracted/<archive_stem>`.
    Returns list of extracted folder paths.
    """
    object_dir = object_dir.resolve()
    extracted_root = object_dir / "extracted"
    extracted_root.mkdir(parents=True, exist_ok=True)

    archive_paths = sorted(
        [
            p
            for p in object_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".zip", ".rar"}
        ]
    )

    if selected_archives is not None:
        selected = set(selected_archives)
        archive_paths = [p for p in archive_paths if p.name in selected]

    out_dirs: List[Path] = []
    for archive in archive_paths:
        out_dir = extracted_root / archive.stem
        if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
            out_dirs.append(out_dir)
            continue
        out_dir.mkdir(parents=True, exist_ok=True)

        if archive.suffix.lower() == ".zip":
            _extract_zip(archive, out_dir)
        else:
            _extract_rar(archive, out_dir)

        out_dirs.append(out_dir)

    return out_dirs

