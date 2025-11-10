from __future__ import annotations

import sys
from pathlib import Path

import h5py
import meshio
import numpy as np
import re

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


def parse_materials(comm_text: str) -> list[tuple[str, float, float]]:
    """Extract (name, E, NU) tuples from the Code_Aster command file."""
    pattern = re.compile(
        r"(\w+)\s*=\s*DEFI_MATERIAU\(\s*ELAS=_F\(\s*E=([0-9eE+.\-]+),\s*NU=([0-9eE+.\-]+)\)\s*\)",
        re.DOTALL,
    )
    matches = pattern.findall(comm_text)
    if not matches:
        raise ValueError("No DEFI_MATERIAU blocks found in .comm file.")

    return [(name, float(E), float(nu)) for name, E, nu in matches]


def extract_function_body(text: str, func_name: str) -> str:
    """Return the inner text of func_name(...) handling nested parentheses."""
    target = f"{func_name}("
    start = text.find(target)
    if start == -1:
        raise ValueError(f"{func_name} call not found in .comm file.")

    idx = start + len(target)
    depth = 1
    body_chars: list[str] = []

    while idx < len(text) and depth > 0:
        char = text[idx]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if depth > 0:
            body_chars.append(char)
        idx += 1

    if depth != 0:
        raise ValueError(f"Unbalanced parentheses while parsing {func_name}.")

    return "".join(body_chars)


def _extract_f_blocks(text: str) -> list[str]:
    """Return the inner text of each `_F(...)` fragment."""
    blocks: list[str] = []
    search_pos = 0
    while True:
        start = text.find("_F(", search_pos)
        if start == -1:
            break
        idx = start + len("_F(")
        depth = 1
        block_chars: list[str] = []
        while idx < len(text) and depth > 0:
            char = text[idx]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            if depth > 0:
                block_chars.append(char)
            idx += 1
        if depth != 0:
            raise ValueError("Unbalanced parentheses while parsing _F block.")
        blocks.append("".join(block_chars))
        search_pos = idx
    return blocks


def parse_group_material_assignments(comm_text: str) -> dict[str, str]:
    """Map mesh group names (GROUP_MA) to material names from AFFE_MATERIAU."""
    affe_body = extract_function_body(comm_text, "AFFE_MATERIAU")
    mapping: dict[str, str] = {}

    for block in _extract_f_blocks(affe_body):
        if "GROUP_MA" not in block or "MATER" not in block:
            continue

        group_match = re.search(r"GROUP_MA\s*=\s*\((.*?)\)", block, re.DOTALL)
        mater_match = re.search(r"MATER\s*=\s*\((.*?)\)", block, re.DOTALL)
        if not (group_match and mater_match):
            continue

        group_names = re.findall(r"'([^']+)'", group_match.group(1))
        mater_name_match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)", mater_match.group(1))
        if not (group_names and mater_name_match):
            continue

        mater_name = mater_name_match.group(1)
        for group in group_names:
            mapping[group] = mater_name

    if not mapping:
        raise ValueError("No GROUP_MA to MATER assignments found in AFFE_MATERIAU.")

    return mapping


def load_family_name_map(med_path: Path) -> dict[int, str]:
    """Read MED family identifiers and their group names via h5py."""
    mapping: dict[int, str] = {}
    with h5py.File(med_path, "r") as handle:
        eleme = handle.get("FAS/Mesh_1/ELEME")
        if eleme is None:
            return mapping

        for fam_key in eleme.keys():
            match = re.match(r"FAM_(-?\d+)_", fam_key)
            if not match:
                continue
            family_id = int(match.group(1))
            name_dataset = eleme[fam_key]["GRO"]["NOM"][...]
            # Convert int8 array into ASCII string (stop at nulls).
            chars = name_dataset[0]
            group_name = "".join(chr(c) for c in chars if c != 0).strip()
            mapping[family_id] = group_name

    return mapping


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


def build_elem_matrix(mesh: meshio.Mesh, tag_to_material: dict[int, int]) -> np.ndarray:
    """Create the elem matrix [type_id, material_row, n1, n2, n3, n4]."""
    block_index, block = locate_volume_block(mesh)
    tags = extract_cell_tags(mesh, block_index)

    connectivity = np.asarray(block.data, dtype=int) + 1  # convert to 1-based ids
    if connectivity.shape[0] != tags.shape[0]:
        raise ValueError("Mismatch between element tags and connectivity lengths.")

    mapped_tags = np.full_like(tags, fill_value=-1, dtype=int)
    for family_id, mat_idx in tag_to_material.items():
        mapped_tags[tags == family_id] = mat_idx

    if np.any(mapped_tags < 0):
        missing = np.unique(tags[mapped_tags < 0])
        raise ValueError(
            f"No material mapping found for family ids: {', '.join(map(str, missing))}"
        )

    elem = np.empty((connectivity.shape[0], 6), dtype=int)
    elem[:, 0] = 1  # element type id (tetra4)
    elem[:, 1] = mapped_tags
    elem[:, 2:] = connectivity
    return elem


def build_tag_to_material_index(
    med_path: Path,
    material_rows: list[tuple[str, float, float]],
    group_assignments: dict[str, str],
) -> dict[int, int]:
    """Return mapping from MED family id to `mater` row index (1-based)."""
    family_map = load_family_name_map(med_path)
    material_lookup = {name: idx + 1 for idx, (name, _, _) in enumerate(material_rows)}
    tag_to_material: dict[int, int] = {}

    for family_id, group_name in family_map.items():
        material_name = group_assignments.get(group_name)
        if not material_name:
            continue
        if material_name not in material_lookup:
            raise ValueError(f"Material '{material_name}' referenced by group '{group_name}' is undefined.")
        tag_to_material[family_id] = material_lookup[material_name]

    if not tag_to_material:
        raise ValueError("Failed to build any material mappings from MED groups.")

    return tag_to_material


def main() -> None:
    target_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("Input")

    try:
        comm_path, med_path = locate_case_files(target_dir)
    except FileNotFoundError as exc:
        sys.exit(str(exc))

    comm_text = comm_path.read_text()

    mesh = load_salome_mesh(med_path)
    node = np.array(mesh.points, dtype=float)
    materials = parse_materials(comm_text)
    group_assignments = parse_group_material_assignments(comm_text)
    tag_to_material = build_tag_to_material_index(med_path, materials, group_assignments)
    elem = build_elem_matrix(mesh, tag_to_material)
    mater = np.array([[E, nu] for _, E, nu in materials], dtype=float)

    print(f"Found command file : {comm_path.name}")
    print(f"Found mesh file    : {med_path.name}")
    print(f"Mesh summary       : {len(mesh.points)} points, {len(mesh.cells)} cell blocks")
    print(f"Node matrix shape  : {node.shape}")
    print(f"Element matrix shape: {elem.shape}")
    print(f"Material matrix shape: {mater.shape}")
    for (name, _, _), row in zip(materials, mater):
        print(f"  {name}: E={row[0]}, nu={row[1]}")


if __name__ == "__main__":
    main()
