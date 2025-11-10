from __future__ import annotations

import sys
from pathlib import Path

import meshio
import numpy as np

VOLUME_CELL_TYPES = {"tetra", "tetra4", "tet10", "tetra10"}


def locate_case_files(input_dir: Path) -> tuple[Path, Path]:
    """
    Locate the first .comm and .med files in the target directory.

    Raises:
        FileNotFoundError: if the directory or required files are missing.
    """
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory '{input_dir}' does not exist.")

    comm_files = sorted(input_dir.glob("*.comm"))
    med_files = sorted(input_dir.glob("*.med"))

    if not comm_files:
        raise FileNotFoundError(f"No .comm files found inside '{input_dir}'.")
    if not med_files:
        raise FileNotFoundError(f"No .med files found inside '{input_dir}'.")

    return comm_files[0], med_files[0]


def load_salome_mesh(med_path: Path) -> meshio.Mesh:
    """Read a Salomé-Meca .med mesh using meshio."""
    return meshio.read(med_path)


def locate_volume_block(mesh: meshio.Mesh) -> tuple[int, meshio.CellBlock]:
    """Return the first volumetric cell block (tetrahedral)."""
    for idx, block in enumerate(mesh.cells):
        if block.type in VOLUME_CELL_TYPES:
            if block.data.shape[1] != 4:
                raise ValueError(
                    f"Expected 4-node tets, got {block.data.shape[1]} nodes per cell."
                )
            return idx, block

    raise ValueError("No tetrahedral cell block found in MED file.")


def extract_cell_tags(mesh: meshio.Mesh, block_index: int) -> np.ndarray:
    """Fetch per-element group identifiers matching the requested block."""
    if "cell_tags" in mesh.cell_data:
        tags = mesh.cell_data["cell_tags"][block_index]
        return np.asarray(tags, dtype=int)

    for data_list in mesh.cell_data.values():
        if block_index < len(data_list):
            tags = data_list[block_index]
            return np.asarray(tags, dtype=int)

    raise ValueError("No cell tags found for tetrahedral block.")


def build_elem_matrix(mesh: meshio.Mesh) -> np.ndarray:
    """Create the elem matrix [type_id, group_id, n1, n2, n3, n4]."""
    block_index, block = locate_volume_block(mesh)
    tags = extract_cell_tags(mesh, block_index)

    connectivity = np.asarray(block.data, dtype=int) + 1  # convert to 1-based ids
    if connectivity.shape[0] != tags.shape[0]:
        raise ValueError("Mismatch between element tags and connectivity lengths.")

    elem = np.empty((connectivity.shape[0], 6), dtype=int)
    elem[:, 0] = 1  # element type id (tetra4)
    elem[:, 1] = tags
    elem[:, 2:] = connectivity
    return elem


def main() -> None:
    target_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("Input")

    try:
        comm_path, med_path = locate_case_files(target_dir)
    except FileNotFoundError as exc:
        sys.exit(str(exc))

    mesh = load_salome_mesh(med_path)
    node = np.array(mesh.points, dtype=float)
    elem = build_elem_matrix(mesh)

    print(f"Found command file : {comm_path.name}")
    print(f"Found mesh file    : {med_path.name}")
    print(f"Mesh summary       : {len(mesh.points)} points, {len(mesh.cells)} cell blocks")
    print(f"Node matrix shape  : {node.shape}")
    print(f"Element matrix shape: {elem.shape}")


if __name__ == "__main__":
    main()
