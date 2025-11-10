# SalomeToPython

CLI that converts a Salomé-Meca MED mesh plus its matching Code_Aster `.comm` file into NumPy-friendly Python arrays (`node`, `elem`, `mater`, `pdof`, `nodf`, etc.).

## Installation

```bash
python -m pip install --user --break-system-packages .
```

This installs the console script `SalomeToPython`. (In constrained environments you can also run it via `python main.py ...` without installing.)

## Usage

```bash
SalomeToPython [-m] [-b] [-o output.py] path/to/case.med path/to/case.comm
```

- `-m`, `--mater` &mdash; include the material matrix parsed from `DEFI_MATERIAU` (`AFFE_MATERIAU` must map groups to materials).
- `-b`, `--boundary` &mdash; include prescribed DOFs (`pdof`) and nodal loads (`nodf`) derived from `DDL_IMPO` / `FORCE_FACE`.
- `-o`, `--output` &mdash; optional destination file (defaults to `<case_folder>.py`).

Without `-m` or `-b`, the generator writes only `node`, `elem`, `eltp`, and `bc_method`, skipping the more expensive parsing steps automatically.

## Development

Install deps once (if you are not using the packaged CLI):

```bash
python -m pip install --break-system-packages meshio h5py
```

Then run the tool directly:

```bash
python main.py -m -b Input/DoubleCubeCase.med Input/DoubleCubeCase.comm
```
