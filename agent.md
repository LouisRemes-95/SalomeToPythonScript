# Agent Notes

This repository contains a small CLI, `Salome2Py`, that converts a Salomé‑Meca `.med` mesh plus its matching Code_Aster `.comm` file into NumPy‑friendly Python arrays (`node`, `elem`, and optionally `mater`, `pdof`, `nodf`).

## Directory Layout

- `Input/`  
  Holds example Salomé‑Meca / Code_Aster cases. Each case is a pair:
  - `*.med` mesh exported from Salomé‑Meca
  - `*.comm` command file defining materials, BCs, loads, etc.

- `Output/`  
  Holds generated Python results for example inputs.
  - If an input example in `Input/` works, its generated `.py` is stored here.
  - `Output/input_example.py` is **not** produced by the CLI; it is a hand‑curated “perfect output” reference for what the arrays should look like.

## CLI Behavior (main.py)

`main.py` is the implementation of the `Salome2Py` CLI described in `README.md`.

Given `case.med case.comm`, the generator:

- Always writes:
  - `node`: `nnode x 4` array `[id, x, y, z]`
  - `elem`: `nelem x 6` array `[id, n1, n2, n3, n4, mat_or_family]`
  - `eltp`: element type id (currently fixed to `1` for tetra4)
  - `bc_method`: string describing BC handling

- With `-m/--mater`:
  - Parses `DEFI_MATERIAU` blocks to build `mater` as `[E, NU]` per material.
  - Maps MED family ids to material rows using `AFFE_MATERIAU` group assignments.

- With `-b/--boundary`:
  - Parses `AFFE_CHAR_MECA` to build:
    - `pdof`: prescribed DOFs per node from `DDL_IMPO`
    - `nodf`: nodal loads from `FORCE_FACE` distributed by triangle areas

Output defaults to `<case_dir>.py` in the current working directory; override with `-o`.

## Runtime Notes

- MED reading needs `meshio` + `h5py`:
  ```bash
  python -m pip install --break-system-packages meshio h5py
  ```
- Typical run:
  ```bash
  python main.py -m -b Input/DoubleCubeCase.med Input/DoubleCubeCase.comm
  ```

## What to Update When Adding New Examples

1. Place new `.med`/`.comm` pairs in `Input/`.
2. Run `Salome2Py` on them.
3. Store the generated `.py` in `Output/` if it matches expectations.
4. Use `Output/input_example.py` as the style/structure benchmark.

## Compatibility Expectation

When extending or fixing the CLI, adjust parsing and generation **smartly** so that:

- New or previously unsupported inputs (with no `Output/` yet) can be converted successfully.
- Existing inputs that are already shown to work continue to produce the same arrays and shapes (backward compatible), unless a deliberate breaking change is agreed on.
