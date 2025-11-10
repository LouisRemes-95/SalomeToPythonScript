# Salome2Py

CLI that converts a Salomé-Meca MED mesh plus its matching Code_Aster `.comm` file into NumPy-friendly Python arrays (`node`, `elem`, `mater`, `pdof`, `nodf`, etc.).

## Installation (pipx)

`pipx` keeps console tools isolated per-virtualenv and works the same on Linux, macOS, and Windows.

1. **Install pipx**
   - Linux/macOS:
     ```bash
     python3 -m pip install --user pipx
     python3 -m pipx ensurepath
     ```
   - Windows (PowerShell):
     ```powershell
     py -m pip install --user pipx
     py -m pipx ensurepath
     ```
   Reload the shell so the `pipx` path is active.

2. **Install Salome2Py**
   - From PyPI (once published):
     ```bash
     pipx install Salome2Py
     ```
   - From a local clone:
     ```bash
     pipx install .
     ```

This exposes the `Salome2Py` command globally without polluting your base interpreter. (You can still run `python main.py ...` directly if you prefer.)

## Usage

```bash
Salome2Py [-m] [-b] [-o output.py] path/to/case.med path/to/case.comm
```

- `-m`, `--mater` &mdash; include the material matrix parsed from `DEFI_MATERIAU` (`AFFE_MATERIAU` must map groups to materials).
- `-b`, `--boundary` &mdash; include prescribed DOFs (`pdof`) and nodal loads (`nodf`) derived from `DDL_IMPO` / `FORCE_FACE`.
- `-o`, `--output` &mdash; optional destination file (defaults to `<case_folder>.py`).

By default, the generated Python file is written to the directory where you invoke the command (e.g., running inside `/tmp` creates `/tmp/input.py` for an `Input/` case). Use `-o` to override the path.

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
