from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import meshio
import numpy as np
import re

VOLUME_CELL_TYPES = {"tetra", "tetra4", "tet10", "tetra10"}
TRIANGLE_CELL_TYPES = {"triangle", "triangle6"}
DISPLACEMENT_DIRECTIONS = {"DX": 1, "DY": 2, "DZ": 3}
FORCE_DIRECTIONS = {"FX": 1, "FY": 2, "FZ": 3}
# Output formatting helpers
FLOAT_TOL = 1e-12


def format_float(value: float) -> str:
    """Format floats without trailing zeros."""
    val = float(value)
    if abs(val) < FLOAT_TOL:
        val = 0.0
    return f"{val:.12g}"


def format_int(value: int) -> str:
    """Format integer values."""
    return str(int(value))


def format_matrix_literal(array: np.ndarray, value_formatter) -> str:
    """Return a Python list literal for a 2D array."""
    if array.size == 0:
        return "[]"

    rows = []
    for row in array.tolist():
        values = ", ".join(value_formatter(val) for val in row)
        rows.append(f"    [{values}]")

    return "[\n" + ",\n".join(rows) + "\n]"


def array_assignment(
    var_name: str,
    array: np.ndarray,
    dtype_str: str,
    value_formatter,
) -> str:
    """Build a numpy array assignment string."""
    if array.size == 0:
        shape = ", ".join(str(dim) for dim in array.shape)
        return f"{var_name} = np.zeros(({shape}), dtype={dtype_str})"

    literal = format_matrix_literal(array, value_formatter)
    return f"{var_name} = np.array({literal}, dtype={dtype_str})"


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


def build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="Salome2Py",
        description="Convert Salomé-Meca MED + Code_Aster .comm files into NumPy-friendly Python data.",
    )
    parser.add_argument(
        "med_path",
        type=Path,
        help="Path to the MED mesh file exported from Salomé-Meca.",
    )
    parser.add_argument(
        "comm_path",
        type=Path,
        help="Path to the matching Code_Aster .comm file.",
    )
    parser.add_argument(
        "-m",
        "--mater",
        action="store_true",
        help="Include the material parameter matrix (mater) in the generated Python file.",
    )
    parser.add_argument(
        "-b",
        "--boundary",
        action="store_true",
        help="Include boundary data (pdof + nodf) in the generated Python file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional path for the generated Python file (defaults to <case_dir>.py next to the MED directory).",
    )
    return parser


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


def _extract_parenthesized(text: str, start_idx: int) -> tuple[str, int]:
    """Return substring inside parentheses starting from start_idx."""
    depth = 1
    chars: list[str] = []
    idx = start_idx

    while idx < len(text) and depth > 0:
        char = text[idx]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if depth > 0:
            chars.append(char)
        idx += 1

    if depth != 0:
        raise ValueError("Unbalanced parentheses while parsing segment.")

    return "".join(chars), idx


def _extract_f_blocks(text: str) -> list[str]:
    """Return the inner text of each `_F(...)` fragment."""
    blocks: list[str] = []
    search_pos = 0
    while True:
        start = text.find("_F(", search_pos)
        if start == -1:
            break
        body, end_idx = _extract_parenthesized(text, start + len("_F("))
        blocks.append(body)
        search_pos = end_idx
    return blocks


def _extract_keyword_blocks(text: str, keyword: str) -> list[str]:
    """Return `_F(...)` fragments that follow `keyword=`."""
    blocks: list[str] = []
    token = f"{keyword}="
    search_pos = 0

    while True:
        start = text.find(token, search_pos)
        if start == -1:
            break
        idx = start + len(token)
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text) or not text.startswith("_F(", idx):
            search_pos = idx
            continue
        body, end_idx = _extract_parenthesized(text, idx + len("_F("))
        blocks.append(body)
        search_pos = end_idx

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


def parse_ddl_impo_blocks(comm_text: str) -> list[dict[str, object]]:
    """Parse DDL_IMPO entries from AFFE_CHAR_MECA."""
    try:
        body = extract_function_body(comm_text, "AFFE_CHAR_MECA")
    except ValueError:
        return []
    blocks: list[dict[str, object]] = []

    for chunk in _extract_keyword_blocks(body, "DDL_IMPO"):
        group_match = re.search(r"GROUP_MA\s*=\s*\((.*?)\)", chunk, re.DOTALL)
        if not group_match:
            continue
        groups = re.findall(r"'([^']+)'", group_match.group(1))
        if not groups:
            continue

        comps: dict[str, float] = {}
        for comp_name in DISPLACEMENT_DIRECTIONS:
            comp_match = re.search(rf"{comp_name}\s*=\s*([0-9eE+.\-]+)", chunk)
            if comp_match:
                comps[comp_name] = float(comp_match.group(1))

        if not comps:
            continue

        blocks.append({"groups": groups, "components": comps})

    return blocks


def parse_force_face_blocks(comm_text: str) -> list[dict[str, object]]:
    """Parse FORCE_FACE entries from AFFE_CHAR_MECA."""
    try:
        body = extract_function_body(comm_text, "AFFE_CHAR_MECA")
    except ValueError:
        return []
    blocks: list[dict[str, object]] = []

    for chunk in _extract_keyword_blocks(body, "FORCE_FACE"):
        group_match = re.search(r"GROUP_MA\s*=\s*\((.*?)\)", chunk, re.DOTALL)
        if not group_match:
            continue
        groups = re.findall(r"'([^']+)'", group_match.group(1))
        if not groups:
            continue

        comps: dict[str, float] = {}
        for comp_name in FORCE_DIRECTIONS:
            comp_match = re.search(rf"{comp_name}\s*=\s*([0-9eE+.\-]+)", chunk)
            if comp_match:
                comps[comp_name] = float(comp_match.group(1))

        if not comps:
            continue

        blocks.append({"groups": groups, "components": comps})

    return blocks


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


def build_elem_matrix(
    mesh: meshio.Mesh, tag_to_material: dict[int, int] | None = None
) -> np.ndarray:
    """Create the elem matrix [type_id, material_or_tag, n1, n2, n3, n4]."""
    block_index, block = locate_volume_block(mesh)
    tags = extract_cell_tags(mesh, block_index)

    connectivity = np.asarray(block.data, dtype=int) + 1  # convert to 1-based ids
    if connectivity.shape[0] != tags.shape[0]:
        raise ValueError("Mismatch between element tags and connectivity lengths.")

    if tag_to_material:
        mapped_tags = np.full_like(tags, fill_value=-1, dtype=int)
        for family_id, mat_idx in tag_to_material.items():
            mapped_tags[tags == family_id] = mat_idx

        if np.any(mapped_tags < 0):
            missing = np.unique(tags[mapped_tags < 0])
            raise ValueError(
                f"No material mapping found for family ids: {', '.join(map(str, missing))}"
            )
    else:
        mapped_tags = tags

    elem = np.empty((connectivity.shape[0], 6), dtype=int)
    elem[:, 0] = 1  # element type id (tetra4)
    elem[:, 1] = mapped_tags
    elem[:, 2:] = connectivity
    return elem


def build_tag_to_material_index(
    family_map: dict[int, str],
    material_rows: list[tuple[str, float, float]],
    group_assignments: dict[str, str],
) -> dict[int, int]:
    """Return mapping from MED family id to `mater` row index (1-based)."""
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


def build_group_node_and_triangle_maps(
    mesh: meshio.Mesh, family_map: dict[int, str]
) -> tuple[dict[str, set[int]], dict[str, list[np.ndarray]]]:
    """Collect node indices and triangle connectivity per group."""
    group_nodes: dict[str, set[int]] = defaultdict(set)
    triangle_groups: dict[str, list[np.ndarray]] = defaultdict(list)

    if "cell_tags" not in mesh.cell_data:
        return group_nodes, triangle_groups

    tags_all = mesh.cell_data["cell_tags"]
    for idx, block in enumerate(mesh.cells):
        if idx >= len(tags_all):
            continue

        tags = np.asarray(tags_all[idx], dtype=int)
        if tags.size == 0:
            continue

        data = np.asarray(block.data, dtype=int)
        for family_id in np.unique(tags):
            group_name = family_map.get(family_id)
            if not group_name:
                continue
            mask = tags == family_id
            selected = data[mask]
            if selected.size == 0:
                continue
            group_nodes[group_name].update(selected.ravel().tolist())
            if block.type in TRIANGLE_CELL_TYPES:
                triangle_groups[group_name].append(selected)

    return group_nodes, triangle_groups


def build_pdof_array(
    group_nodes: dict[str, set[int]], ddl_blocks: list[dict[str, object]]
) -> np.ndarray:
    """Convert DDL_IMPO definitions to pdof array."""
    entries: dict[tuple[int, int], float] = {}

    for block in ddl_blocks:
        groups = block["groups"]
        components = block["components"]
        for group in groups:
            nodes = sorted(group_nodes.get(group, set()))
            if not nodes:
                continue
            for comp_name, direction in DISPLACEMENT_DIRECTIONS.items():
                value = components.get(comp_name)
                if value is None:
                    continue
                for node in nodes:
                    entries[(node + 1, direction)] = float(value)

    if not entries:
        return np.zeros((0, 3), dtype=float)

    rows = [
        [node, direction, value]
        for (node, direction), value in sorted(entries.items())
    ]
    return np.array(rows, dtype=float)


def triangle_area(points: np.ndarray) -> float:
    """Return area of triangle defined by three points."""
    vec1 = points[1] - points[0]
    vec2 = points[2] - points[0]
    return 0.5 * np.linalg.norm(np.cross(vec1, vec2))


def build_nodal_loads(
    node_coords: np.ndarray,
    triangle_groups: dict[str, list[np.ndarray]],
    force_blocks: list[dict[str, object]],
) -> np.ndarray:
    """Convert FORCE_FACE traction blocks to nodal force array."""
    loads: dict[tuple[int, int], float] = {}

    for block in force_blocks:
        components = block["components"]
        traction = np.array(
            [
                components.get("FX", 0.0),
                components.get("FY", 0.0),
                components.get("FZ", 0.0),
            ],
            dtype=float,
        )
        if np.allclose(traction, 0.0):
            continue

        for group in block["groups"]:
            tri_sets = triangle_groups.get(group, [])
            for tris in tri_sets:
                for tri in tris:
                    coords = node_coords[tri]
                    area = triangle_area(coords)
                    if area == 0.0:
                        continue
                    nodal_force = traction * (area / 3.0)
                    for node_idx in tri:
                        node_id = node_idx + 1
                        for direction, value in enumerate(nodal_force, start=1):
                            if value == 0.0:
                                continue
                            key = (node_id, direction)
                            loads[key] = loads.get(key, 0.0) + float(value)

    if not loads:
        return np.zeros((0, 3), dtype=float)

    rows = [
        [node, direction, value]
        for (node, direction), value in sorted(loads.items())
    ]
    return np.array(rows, dtype=float)


def write_case_file(
    case_dir: Path,
    node: np.ndarray,
    elem: np.ndarray,
    pdof: np.ndarray | None,
    nodf: np.ndarray | None,
    mater: np.ndarray | None,
    material_labels: list[str],
    include_mater: bool,
    include_boundary: bool,
    explicit_output: Path | None = None,
) -> Path:
    """Persist extracted data into a Python module."""
    case_name = case_dir.name.lower()
    output_path = explicit_output or (Path.cwd() / f"{case_name}.py")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    elem_export = elem.copy()
    if elem_export.shape[1] >= 6:
        elem_export[:, [4, 5]] = elem_export[:, [5, 4]]

    lines = [
        f"# {output_path.name}",
        "import numpy as np",
        "",
        "# nodal coordinates [x, y, z]",
        array_assignment("node", node, "float", format_float),
        "",
        "# mesh connectivity [type#, material#, node1 node2 node3 node4]",
        array_assignment("elem", elem_export, "int", format_int),
    ]

    if include_boundary and pdof is not None:
        lines.extend(
            [
                "",
                "# prescribed degrees of freedom (displacements) [node# dir value]",
                array_assignment("pdof", pdof, "float", format_float),
                "",
                "# applied nodal loads [node# dir value]",
                array_assignment("nodf", nodf, "float", format_float),
            ]
        )

    lines.extend(
        [
            "",
            "# finite element type table",
            'eltp = {1: "tetra4"}',
        ]
    )

    if include_mater and mater is not None:
        lines.extend(
            [
                "",
                "# material parameter matrix (rows = materials)",
                array_assignment("mater", mater, "float", format_float),
            ]
        )

    if include_mater and material_labels:
        label_str = ", ".join(f'"{label}"' for label in material_labels)
        lines.extend(
            [
                "",
                "# material row labels (matches `mater` order)",
                f"material_labels = [{label_str}]",
            ]
        )

    lines.extend(
        [
            "",
            "# Boundary condition method",
            "# < 0 direct, >= 0 penalty",
            "bc_method = -1",
            "",
        ]
    )

    output_path.write_text("\n".join(lines))
    return output_path
def main(argv: list[str] | None = None) -> None:
    parser = build_cli_parser()
    args = parser.parse_args(argv)

    med_path = args.med_path
    comm_path = args.comm_path

    if not med_path.is_file():
        parser.error(f"MED file not found: {med_path}")
    if not comm_path.is_file():
        parser.error(f".comm file not found: {comm_path}")

    include_mater = args.mater
    include_boundary = args.boundary

    family_map = load_family_name_map(med_path)
    mesh = load_salome_mesh(med_path)
    node = np.array(mesh.points, dtype=float)

    materials: list[tuple[str, float, float]] = []
    material_labels: list[str] = []
    mater: np.ndarray | None = None
    tag_to_material: dict[int, int] | None = None

    if include_mater:
        comm_text = comm_path.read_text()
        materials = parse_materials(comm_text)
        material_labels = [name for name, _, _ in materials]
        group_assignments = parse_group_material_assignments(comm_text)
        tag_to_material = build_tag_to_material_index(family_map, materials, group_assignments)
        mater = np.array([[E, nu] for _, E, nu in materials], dtype=float)
    else:
        comm_text = comm_path.read_text() if include_boundary else ""

    elem = build_elem_matrix(mesh, tag_to_material)

    pdof: np.ndarray | None = None
    nodf: np.ndarray | None = None

    if include_boundary:
        if not comm_text:
            comm_text = comm_path.read_text()
        group_nodes, triangle_groups = build_group_node_and_triangle_maps(mesh, family_map)
        ddl_blocks = parse_ddl_impo_blocks(comm_text)
        pdof = build_pdof_array(group_nodes, ddl_blocks)
        force_blocks = parse_force_face_blocks(comm_text)
        nodf = build_nodal_loads(node, triangle_groups, force_blocks)

    output_path = write_case_file(
        med_path.parent,
        node,
        elem,
        pdof,
        nodf,
        mater,
        material_labels,
        include_mater,
        include_boundary,
        args.output,
    )

    print(f"Found command file : {comm_path}")
    print(f"Found mesh file    : {med_path}")
    print(f"Mesh summary       : {len(mesh.points)} points, {len(mesh.cells)} cell blocks")
    print(f"Node matrix shape  : {node.shape}")
    print(f"Element matrix shape: {elem.shape}")

    if include_mater and mater is not None:
        print(f"Material matrix shape: {mater.shape}")
        for (name, _, _), row in zip(materials, mater):
            print(f"  {name}: E={row[0]}, nu={row[1]}")

    if include_boundary and pdof is not None and nodf is not None:
        print(f"Prescribed DOF shape: {pdof.shape}")
        print(f"Nodal load shape     : {nodf.shape}")

    print(f"Wrote case file      : {output_path}")


if __name__ == "__main__":
    main()
